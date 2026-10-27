use duck_control::profile::{Joint, PolicyLayout, RobotProfile};
use duck_control::safety::{Limit, Safety, SafetyConfig};
use duck_control::{FakeIo, ImuData, RobotIo};

fn description(n: usize) -> RobotProfile {
    RobotProfile {
        name: format!("test-{n}"),
        joints: (0..n)
            .map(|i| Joint {
                name: format!("joint_{i}"),
                id: i as u8 + 1,
                home: 0.0,
                min: -0.5,
                max: 0.5,
                action_scale: 0.25,
            })
            .collect(),
        policy: PolicyLayout {
            joints: (0..n).map(|i| format!("joint_{i}")).collect(),
            commands: vec!["vx".into(), "vy".into(), "yaw_rate".into()],
        },
    }
}

fn exercise<const N: usize>() {
    let profile = description(N).validate().unwrap();
    let mut safety =
        Safety::for_profile(FakeIo::<N>::new(), SafetyConfig::default(), &profile).unwrap();
    let mut obs = vec![0.0; profile.observation_len()];
    let sensors = safety.read().unwrap();
    profile
        .observation(
            &sensors.imu,
            &sensors.positions,
            &sensors.velocities,
            &vec![0.0; N],
            &[0.1, 0.0, 0.0],
            &mut obs,
        )
        .unwrap();
    assert_eq!(obs.len(), 9 + 3 * N);
    let mut targets = [0.0; N];
    profile
        .targets(&vec![4.0; N], &[0.0; N], &mut targets)
        .unwrap();
    assert!(
        safety
            .apply(targets, sensors.positions, 100)
            .unwrap()
            .limited_by(Limit::Range)
    );
    assert_eq!(safety.read().unwrap().positions, [0.5; N]);
    // A NaN action must use the hold path, including the mechanical limits.
    safety.apply([f64::NAN; N], [-1.0; N], 100).unwrap();
    assert_eq!(safety.read().unwrap().positions, [-0.5; N]);
    assert!(safety.apply([f64::NAN; N], [f64::NAN; N], 100).is_err());
    assert_eq!(safety.read().unwrap().positions, [-0.5; N]);
}

#[test]
fn existing_safety_and_io_work_below_and_above_fifteen_joints() {
    exercise::<1>();
    exercise::<4>();
    exercise::<15>();
    exercise::<24>();
}

#[test]
fn joint_order_is_explicit_and_uncontrolled_joints_are_preserved() {
    let mut raw = description(4);
    raw.policy.joints = vec!["joint_3".into(), "joint_1".into()];
    raw.joints[3].home = 0.1;
    raw.joints[3].action_scale = -0.25;
    let profile = raw.validate().unwrap();
    let mut obs = vec![0.0; profile.observation_len()];
    profile
        .observation(
            &ImuData::default(),
            &[0.0, 0.2, 0.3, 0.4],
            &[1.0, 2.0, 3.0, 4.0],
            &[0.6, 0.7],
            &[0.8, 0.9, 1.0],
            &mut obs,
        )
        .unwrap();
    assert_eq!(&obs[6..], &[0.3, 0.2, 4.0, 2.0, 0.6, 0.7, 0.8, 0.9, 1.0]);
    let mut targets = [0.0; 4];
    profile
        .targets(&[1.0, 2.0], &[0.11, 0.22, 0.33, 0.44], &mut targets)
        .unwrap();
    assert_eq!(targets, [0.11, 0.5, 0.33, -0.15]);
    let saved = targets;
    assert!(
        profile
            .targets(&[f32::NAN, 0.0], &[0.0; 4], &mut targets)
            .is_err()
    );
    assert_eq!(targets, saved);
}

#[test]
fn invalid_profiles_cannot_reach_the_bus() {
    for change in 0..8 {
        let mut profile = description(4);
        match change {
            0 => profile.joints[1].id = profile.joints[0].id,
            1 => profile.joints[0].id = 254,
            2 => profile.joints[0].min = 1.0,
            3 => profile.joints[0].home = f64::NAN,
            4 => profile.policy.joints.push("absent".into()),
            5 => profile.policy.joints.push("joint_0".into()),
            6 => profile.policy.commands.push("vx".into()),
            _ => profile.joints[0].action_scale = 0.0,
        }
        assert!(profile.validate().is_err(), "invalid case {change}");
    }
    let profile = description(4).validate().unwrap();
    assert!(Safety::for_profile(FakeIo::<15>::new(), SafetyConfig::default(), &profile).is_err());
}

#[test]
fn observation_refuses_short_and_non_finite_inputs() {
    let profile = description(4).validate().unwrap();
    let mut obs = vec![0.0; 21];
    for positions in [vec![0.0; 3], vec![f64::NAN; 4], vec![f64::MAX; 4]] {
        assert!(
            profile
                .observation(
                    &ImuData::default(),
                    &positions,
                    &[0.0; 4],
                    &[0.0; 4],
                    &[0.0; 3],
                    &mut obs
                )
                .is_err()
        );
    }
}

#[test]
fn alpha_observation_is_unchanged() {
    use duck_control::{Command, DEFAULT_POSITION, JOINT_IDS, JOINT_NAMES, Observation};
    let mut raw = description(15);
    for (i, joint) in raw.joints.iter_mut().enumerate() {
        joint.name = JOINT_NAMES[i].into();
        joint.id = JOINT_IDS[i];
        joint.home = DEFAULT_POSITION[i];
        joint.min = -3.0;
        joint.max = 3.0;
    }
    raw.policy.joints = JOINT_NAMES
        .iter()
        .filter(|&&n| n != "mouth")
        .map(|&n| n.into())
        .collect();
    raw.policy.commands = (0..13).map(|i| format!("command_{i}")).collect();
    let profile = raw.validate().unwrap();
    let positions = std::array::from_fn(|i| i as f64 / 10.0);
    let velocities = std::array::from_fn(|i| -(i as f64) / 20.0);
    let actions = std::array::from_fn(|i| i as f32 / 30.0);
    let imu = ImuData {
        gyro: [0.1, 0.2, 0.3],
        ..ImuData::default()
    };
    let command = Command {
        twist: [0.1, 0.2, 0.3],
        head: [0.4, 0.5, 0.6, 0.7],
        body: duck_control::obs::BodyPose {
            z: 0.8,
            roll: 0.9,
            pitch: 1.0,
        },
    };
    let original = Observation::build(
        &imu,
        &positions,
        &velocities,
        &DEFAULT_POSITION,
        &actions,
        &command,
    );
    let mut actual = vec![0.0; profile.observation_len()];
    profile
        .observation(
            &imu,
            &positions,
            &velocities,
            &actions,
            &[
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0, 0.0, 0.8, 0.9, 1.0, 0.0,
            ],
            &mut actual,
        )
        .unwrap();
    assert_eq!(actual, original.as_slice());
}

#[test]
fn four_joint_example_is_a_real_validated_contract() {
    let profile = RobotProfile::from_toml(include_str!(
        "../../deploy/profiles/four-joint.example.toml"
    ))
    .unwrap();
    assert_eq!(
        (
            profile.joint_count(),
            profile.observation_len(),
            profile.action_len()
        ),
        (4, 21, 4)
    );
    let text = toml::to_string(profile.description()).unwrap();
    assert!(RobotProfile::from_toml(&text.replace("home =", "typo =")).is_err());
    let mut io = FakeIo::<4>::new();
    assert!(io.torque.is_none());
    io.read().unwrap();
    assert!(io.torque.is_none());
}

#[test]
#[ignore = "requires ONNX Runtime >= 1.23"]
fn different_morphologies_use_the_real_onnx_loader() {
    use duck_control::policy::ProfilePolicy;
    use std::path::Path;
    for n in [4, 24] {
        let profile = description(n).validate().unwrap();
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/fixtures/profile_{n}.onnx"));
        let mut policy = ProfilePolicy::load(&path, &profile).unwrap();
        let mut obs = vec![0.0; profile.observation_len()];
        for i in 0..n {
            obs[6 + i] = i as f32 / 10.0;
        }
        assert_eq!(policy.infer(&obs).unwrap(), obs[6..6 + n]);
        assert!(policy.infer(&obs[..obs.len() - 1]).is_err());
        obs[0] = f32::NAN;
        assert!(policy.infer(&obs).is_err());
        let wrong = description(if n == 4 { 24 } else { 4 }).validate().unwrap();
        assert!(ProfilePolicy::load(&path, &wrong).is_err());
    }
}
