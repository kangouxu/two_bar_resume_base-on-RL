from stable_baselines3 import PPO
from robo1_env import Robo1GetupEnv

model = PPO.load("robo1_getup_ppo", device="cpu")


for pose in ("roll_pos", "roll_neg", "pitch_pos", "pitch_neg"):
    env = Robo1GetupEnv("robo1.xml", fallen_poses=(pose,))
    obs, _ = env.reset(options={"pose": pose})
    best_upright = -1.0
    got_goal = False
    for _ in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        best_upright = max(best_upright, info["upright"])
        got_goal = got_goal or info["goal_pose"]
        if terminated or truncated:
            break
    print(pose, "best_upright:", round(best_upright, 4), "goal_pose:", got_goal)