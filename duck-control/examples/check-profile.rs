//! Validate a morphology without opening hardware or enabling torque.
use duck_control::profile::RobotProfile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: check-profile <profile.toml>")?;
    let profile = RobotProfile::from_toml(&std::fs::read_to_string(path)?)?;
    println!(
        "{}: {} joints, {} observations, {} actions",
        profile.description().name,
        profile.joint_count(),
        profile.observation_len(),
        profile.action_len()
    );
    Ok(())
}
