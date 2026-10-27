//! Validated morphology and policy ordering, independent of a particular robot.
//!
//! See `docs/design/robotd-design.md` for the integration boundary. A profile describes
//! joints and the learned policy contract; it does not imply a supported hardware driver.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::imu::ImuData;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Joint {
    pub name: String,
    pub id: u8,
    pub home: f64,
    pub min: f64,
    pub max: f64,
    /// Radians per unit of policy output, including the joint's direction.
    pub action_scale: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyLayout {
    /// Names in training order, which need not match the physical bus order.
    pub joints: Vec<String>,
    /// Names in the trailing command block, also in training order.
    pub commands: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobotProfile {
    pub name: String,
    pub joints: Vec<Joint>,
    pub policy: PolicyLayout,
}

#[derive(Debug, thiserror::Error)]
#[error("robot profile: {0}")]
pub struct ProfileError(pub String);

/// Immutable after validation, so the action map cannot drift from the joint table.
#[derive(Debug, Clone)]
pub struct ValidatedProfile {
    profile: RobotProfile,
    policy_indices: Vec<usize>,
}

impl RobotProfile {
    pub fn from_toml(text: &str) -> Result<ValidatedProfile, ProfileError> {
        toml::from_str::<Self>(text)
            .map_err(|e| ProfileError(e.to_string()))?
            .validate()
    }

    pub fn validate(self) -> Result<ValidatedProfile, ProfileError> {
        let fail = |s: &str| ProfileError(s.to_owned());
        if self.name.trim().is_empty() || self.joints.is_empty() || self.joints.len() > 253 {
            return Err(fail("name and 1..=253 joints are required"));
        }
        let mut names = HashSet::new();
        let mut ids = HashSet::new();
        for joint in &self.joints {
            if joint.name.trim().is_empty() || !names.insert(joint.name.as_str()) {
                return Err(fail("joint names must be nonempty and unique"));
            }
            if joint.id > 252 || !ids.insert(joint.id) {
                return Err(fail("Dynamixel IDs must be unique and in 0..=252"));
            }
            if ![joint.home, joint.min, joint.max, joint.action_scale]
                .iter()
                .all(|v| v.is_finite())
                || joint.min >= joint.max
                || !(joint.min..=joint.max).contains(&joint.home)
                || joint.action_scale == 0.0
            {
                return Err(ProfileError(format!(
                    "{}: invalid home, limits, or action scale",
                    joint.name
                )));
            }
        }
        if self.policy.joints.is_empty() {
            return Err(fail("policy must control at least one joint"));
        }
        let mut seen = HashSet::new();
        let mut policy_indices = Vec::new();
        for name in &self.policy.joints {
            if !seen.insert(name) {
                return Err(fail("policy joint names must be unique"));
            }
            policy_indices.push(
                self.joints
                    .iter()
                    .position(|j| &j.name == name)
                    .ok_or_else(|| ProfileError(format!("unknown policy joint {name}")))?,
            );
        }
        seen.clear();
        if self.policy.commands.len() > 1024
            || self
                .policy
                .commands
                .iter()
                .any(|name| name.trim().is_empty() || !seen.insert(name))
        {
            return Err(fail(
                "at most 1024 unique, nonempty command names are allowed",
            ));
        }
        Ok(ValidatedProfile {
            profile: self,
            policy_indices,
        })
    }
}

impl ValidatedProfile {
    pub fn description(&self) -> &RobotProfile {
        &self.profile
    }
    pub fn joint_count(&self) -> usize {
        self.profile.joints.len()
    }
    pub fn action_len(&self) -> usize {
        self.policy_indices.len()
    }
    /// Gyro, gravity, relative positions, velocities, previous actions, commands.
    pub fn observation_len(&self) -> usize {
        6 + 3 * self.action_len() + self.profile.policy.commands.len()
    }

    /// Caller-owned storage keeps allocation out of the observation assembly path.
    pub fn observation(
        &self,
        imu: &ImuData,
        positions: &[f64],
        velocities: &[f64],
        last_action: &[f32],
        commands: &[f64],
        out: &mut [f32],
    ) -> Result<(), ProfileError> {
        if positions.len() != self.joint_count()
            || velocities.len() != self.joint_count()
            || last_action.len() != self.action_len()
            || commands.len() != self.profile.policy.commands.len()
            || out.len() != self.observation_len()
        {
            return Err(ProfileError(
                "observation dimensions disagree with profile".into(),
            ));
        }
        if !positions
            .iter()
            .chain(velocities)
            .chain(commands)
            .chain(&imu.gyro)
            .chain(&imu.gravity)
            .all(|v| v.is_finite())
            || !last_action.iter().all(|v| v.is_finite())
        {
            return Err(ProfileError("non-finite observation input".into()));
        }
        let n = self.action_len();
        for (dst, src) in out[..6].iter_mut().zip(imu.gyro.iter().chain(&imu.gravity)) {
            *dst = *src as f32;
        }
        for (slot, &joint) in self.policy_indices.iter().enumerate() {
            out[6 + slot] = (positions[joint] - self.profile.joints[joint].home) as f32;
            out[6 + n + slot] = velocities[joint] as f32;
            out[6 + 2 * n + slot] = last_action[slot];
        }
        for (dst, src) in out[6 + 3 * n..].iter_mut().zip(commands) {
            *dst = *src as f32;
        }
        if !out.iter().all(|v| v.is_finite()) {
            return Err(ProfileError("observation exceeds float32 range".into()));
        }
        Ok(())
    }

    /// Uncontrolled joints retain their requested hold pose; controlled joints use
    /// home + scaled action. The shared safety layer owns enforcement of travel limits.
    pub fn targets(
        &self,
        actions: &[f32],
        hold: &[f64],
        out: &mut [f64],
    ) -> Result<(), ProfileError> {
        if actions.len() != self.action_len()
            || hold.len() != self.joint_count()
            || out.len() != self.joint_count()
        {
            return Err(ProfileError(
                "action dimensions disagree with profile".into(),
            ));
        }
        if !actions.iter().all(|v| v.is_finite()) || !hold.iter().all(|v| v.is_finite()) {
            return Err(ProfileError("non-finite action or hold pose".into()));
        }
        // Validate before touching caller storage; a rejected action cannot leave a
        // partially updated command behind for a caller to accidentally send.
        for (&index, &action) in self.policy_indices.iter().zip(actions) {
            let joint = &self.profile.joints[index];
            if !(joint.home + joint.action_scale * f64::from(action)).is_finite() {
                return Err(ProfileError("scaled action exceeds float64 range".into()));
            }
        }
        out.copy_from_slice(hold);
        for (&index, &action) in self.policy_indices.iter().zip(actions) {
            let joint = &self.profile.joints[index];
            out[index] = joint.home + joint.action_scale * f64::from(action);
        }
        Ok(())
    }

    pub fn limits<const N: usize>(&self) -> Result<[(f64, f64); N], ProfileError> {
        if N != self.joint_count() {
            return Err(ProfileError(format!(
                "backend has {N} joints, profile has {}",
                self.joint_count()
            )));
        }
        Ok(std::array::from_fn(|i| {
            (self.profile.joints[i].min, self.profile.joints[i].max)
        }))
    }
}
