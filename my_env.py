import math
import numpy as np
import gymnasium as gym
import mujoco
from gymnasium import spaces


def roll_pitch_to_quat(roll,pitch):
    cr,sr = math.cos(roll/2),math.sin(roll/2)
    cp,sp = math.cos(pitch/2),math.sin(pitch/2)
    return np.array([cr * cp, sr * cp, cr * sp, -sr * sp], dtype=np.float64)

def quat_to_roll_pitch(q):
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    return roll, pitch

#print(quat_to_roll_pitch([1.0, 0.0, 0.0, 0.0]))  #输入四元数，返回roll和pitch

FALLEN_POSES = {  #有四种姿态
    "roll_pos": (math.pi / 2, 0.0),
    "roll_neg": (-math.pi / 2, 0.0),
    "pitch_pos": (0.0, math.pi / 2),
    "pitch_neg": (0.0, -math.pi / 2),
}

class Robo1GetupEnv(gym.Env):
    def __init__(self, xml_path="robo1.xml"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-math.pi, -math.pi, -1.55, -1.55], dtype=np.float32),
            high=np.array([math.pi, math.pi, 1.55, 1.55], dtype=np.float32),
            dtype=np.float32,
        )
        self.foot_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "foot")
        self.servo1_qpos_id = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "servo1_joint")]
        self.servo2_qpos_id = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "servo2_joint")]
        self.frame_skip = 20       # 一次 step 推 20 个仿真步 = 20ms
        self.target_delta = 0.08   # 动作 -> 目标角增量的缩放
        self.target_limit = 1.55   # 目标角限幅（rad）
        self.settle_steps = 200    # reset 后让身体先稳定下来的步数
        self.max_steps = 700       # 单局步数上限

        self.target = np.zeros(2, dtype=np.float64)   # 两个舵机当前目标角
        self.prev_upright = 0.0                       # 上一步的 upright
        self.step_count = 0                           # 已走步数

# e = Robo1GetupEnv('robo1.xml')
# print(e.action_space.shape, e.observation_space.shape)

    def _get_obs(self):
        roll, pitch = quat_to_roll_pitch(self.data.qpos[3:7])
        return np.array([roll, pitch, self.target[0], self.target[1]],
                        dtype=np.float32)

    def _upright(self):
        xmat = self.data.xmat[self.foot_body_id].reshape(3, 3)
        return float(xmat[2, 2])

    def _set_pose(self, pose_name):
        roll, pitch = FALLEN_POSES[pose_name]
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[0:3] = [0.0, 0.0, 0.08]
        self.data.qpos[3:7] = roll_pitch_to_quat(roll, pitch)
        self.target[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(self.settle_steps):
            self.data.ctrl[:] = 0.0
            mujoco.mj_step(self.model, self.data)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        pose = self.np_random.choice(list(FALLEN_POSES.keys()))
        self.current_pose_name = pose
        self._set_pose(pose)
        self.prev_upright = self._upright()
        return self._get_obs(), {"pose": pose}

# e = Robo1GetupEnv('robo1.xml')
# o, info = e.reset()
# print(info['pose'], o)
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        self.target += action * self.target_delta
        self.target = np.clip(self.target, -self.target_limit, self.target_limit)

        for _ in range(self.frame_skip):
            self.data.ctrl[:] = self.target
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs = self._get_obs()
        upright = self._upright()
        upright_progress = upright - self.prev_upright
        self.prev_upright = upright
        roll, pitch = float(obs[0]), float(obs[1])

        tilt_cost = 0.2 * (roll * roll + pitch * pitch)
        reward = 2.0 * upright + 8.0 * upright_progress - tilt_cost
        truncated = self.step_count >= self.max_steps
        return obs, float(reward), False, truncated, {"upright": upright}


# e = Robo1GetupEnv('robo1.xml'); 
# o,_ = e.reset(); 
# [print(e.step(e.action_space.sample())[1]) for _ in range(5)]

# from stable_baselines3.common.env_checker import check_env
# from my_env import Robo1GetupEnv
# env = Robo1GetupEnv("robo1.xml")
# check_env(env, warn=True)          

# obs, info = env.reset()
# total = 0.0
# for i in range(700):
#     obs, r, terminated, truncated, info = env.step(env.action_space.sample())
#     total += r
#     if i % 100 == 0:
#         print(f"step {i}: upright={info['upright']:.3f}, reward={r:.3f}")
#     if terminated or truncated:
#         break
# print(f"累计 reward: {total:.1f}")