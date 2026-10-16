# 第 1 站：从零敲出一个 RL 环境

> 这一站我们不复制任何现成代码，**一行一行把 `my_env.py` 敲出来**。
> 每敲完一段，都有一个「运行验证」帮你确认没敲错。最终你会得到一个能跑的强化学习环境。
>
> 前提：你在 `RL/` 目录下（旁边有 `robo1.xml` 和 `assets/`），并且已经
> `pip install gymnasium mujoco numpy stable-baselines3`。

---

## 先明确：我们要造什么

一个继承 `gym.Env` 的类，叫 `Robo1GetupEnv`，对外只提供三样东西，RL 算法就靠它们学习：

| 成员 | 作用 |
| --- | --- |
| `observation_space` / `action_space` | 告诉算法"观测/动作长什么样" |
| `reset()` | 开局，回到倒地姿态，返回初始观测 |
| `step(action)` | 走一步，返回 `(新观测, 奖励, ...)` |

整个文件约 120 行。我们分成 8 步敲。

---

## Step 0：建文件，敲导入

新建 `my_env.py`，敲这 5 行：

```python
import math
import numpy as np
import gymnasium as gym
import mujoco
from gymnasium import spaces
```

逐行说：

- `math` —— 算三角函数和弧度。
- `numpy` —— 数组（`np.array`）。
- `gymnasium` —— 环境基类 `gym.Env`。
- `mujoco` —— 物理引擎，加载模型、推进仿真。
- `spaces` —— 定义观测/动作空间的类型。

**运行验证**（确认导入不报错）：

```powershell
python -c "import my_env; print('ok')"
```

预期输出：`ok`

---

## Step 1：敲四元数 ↔ roll/pitch 工具函数

在文件末尾**继续往下敲**：

```python
def roll_pitch_to_quat(roll, pitch):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    return np.array([cr * cp, sr * cp, cr * sp, -sr * sp], dtype=np.float64)


def quat_to_roll_pitch(q):
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    return roll, pitch
```

为什么需要这两个函数：

- MuJoCo 里身体姿态存的是**四元数**（`qpos[3:7]`），但观测要喂给网络的是 **roll/pitch**（IMU 能测的角度）。
- `quat_to_roll_pitch`：四元数 → 两个角度（`_get_obs` 里用）。
- `roll_pitch_to_quat`：两个角度 → 四元数（`reset` 摆倒地姿态时用）。
- 第 9 行的 `max/min` 是**防浮点误差**：`asin` 的入参一旦超出 `[-1,1]` 就会报错。

**运行验证**：

```powershell
python -c "import my_env; print(my_env.quat_to_roll_pitch([1.0, 0.0, 0.0, 0.0]))"
```

预期输出：`(0.0, 0.0)`（四元数 `[1,0,0,0]` 代表"无旋转"）。

---

## Step 2：敲四种倒地姿态

继续往下敲：

```python
FALLEN_POSES = {
    "roll_pos": (math.pi / 2, 0.0),
    "roll_neg": (-math.pi / 2, 0.0),
    "pitch_pos": (0.0, math.pi / 2),
    "pitch_neg": (0.0, -math.pi / 2),
}
```

- 每种倒地姿态用 `(roll, pitch)` 一对角度表示。
- `roll` 是侧倒（绕 x 轴），`pitch` 是前后倒（绕 y 轴）。`±π/2` 就是完全倒向一侧/前后。
- 后面 `reset` 会随机从这四个里挑一个开局。

---

## Step 3：敲类骨架 + `__init__`（本文件最长的一段）

继续往下敲：

```python
class Robo1GetupEnv(gym.Env):
    def __init__(self, xml_path="robo1.xml"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
```

说明：

- `class Robo1GetupEnv(gym.Env)` —— 继承环境基类。
- `from_xml_path` 加载 `robo1.xml`，得到 `model`（**静态的机器人定义**）。
- `MjData(model)` 得到 `data`（**当前这一帧的运行状态**，比如位置、速度）。之后推进仿真都是在动 `data`。

**在同一个 `__init__` 里**，继续往下敲空间定义（注意缩进 8 个空格）：

```python
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-math.pi, -math.pi, -1.55, -1.55], dtype=np.float32),
            high=np.array([math.pi, math.pi, 1.55, 1.55], dtype=np.float32),
            dtype=np.float32,
        )
```

说明：

- 动作空间：2 维连续，每维在 `[-1, 1]`。
- 观测空间：4 维连续。前两维 `roll/pitch` 范围 `±π`；后两维是舵机目标角，范围 `±1.55`（对应关节限位）。

**继续往下敲**（还是 `__init__` 里）：把后面要用到的 MuJoCo id 提前查出来，避免每次现查：

```python
        self.foot_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "foot")
        self.servo1_qpos_id = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "servo1_joint")]
        self.servo2_qpos_id = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "servo2_joint")]
```

说明：

- `mj_name2id`：用**名字**查元素 id（"foot" 是身体，"servo1_joint" 是关节）。
- `jnt_qposadr`：该关节在 `data.qpos` 数组里的**起始下标**。之后读舵机角度就用它。

**继续往下敲**控制参数和状态变量（`__init__` 结尾）：

```python
        self.frame_skip = 20       # 一次 step 推 20 个仿真步 = 20ms
        self.target_delta = 0.08   # 动作 -> 目标角增量的缩放
        self.target_limit = 1.55   # 目标角限幅（rad）
        self.settle_steps = 200    # reset 后让身体先稳定下来的步数
        self.max_steps = 700       # 单局步数上限

        self.target = np.zeros(2, dtype=np.float64)   # 两个舵机当前目标角
        self.prev_upright = 0.0                       # 上一步的 upright
        self.step_count = 0                           # 已走步数
```

**运行验证**（实例化，确认空间和模型加载正常）：

```powershell
python -c "from my_env import Robo1GetupEnv; e = Robo1GetupEnv('robo1.xml'); print(e.action_space.shape, e.observation_space.shape)"
```

预期输出：`(2,) (4,)`

---

## Step 4：敲内部工具方法

继续往下敲（注意：现在开始**缩进回到 4 个空格**，是类的方法了）：

```python
    def _get_obs(self):
        roll, pitch = quat_to_roll_pitch(self.data.qpos[3:7])
        return np.array([roll, pitch, self.target[0], self.target[1]],
                        dtype=np.float32)
```

说明：观测 = 身体姿态 `[roll, pitch]` + 两个舵机目标角 `[target0, target1]`。**为什么要把目标角放进观测？** 因为动作是目标角的"增量"，给定同样姿态、不同目标角对应不同下一步，缺了它策略会乱抖。

继续往下敲：

```python
    def _upright(self):
        xmat = self.data.xmat[self.foot_body_id].reshape(3, 3)
        return float(xmat[2, 2])
```

说明：`xmat` 是 foot 身体的朝向矩阵，`xmat[2,2]` 是"身体 z 轴在世界 z 轴上的投影"——**站直时≈1，躺平时≈0**。这是一个连续的"站得多直"的度量，后面奖励全靠它。

继续往下敲：

```python
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
```

说明（逐块）：

- `qpos[:] = 0` / `qvel[:] = 0`：清空所有位置和速度。
- `qpos[0:3] = [0,0,0.08]`：把身体悬空 8cm。
- `qpos[3:7] = roll_pitch_to_quat(...)`：转到指定倒地姿态。
- `mj_forward`：根据新的 qpos 刷新一次内部状态。
- `settle_steps` 循环：**让身体落地、稳定下来再开始**。如果不做这步，开局第一帧是"自由落体"，会污染学习。

---

## Step 5：敲 `reset`

继续往下敲：

```python
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        pose = self.np_random.choice(list(FALLEN_POSES.keys()))
        self.current_pose_name = pose
        self._set_pose(pose)
        self.prev_upright = self._upright()
        return self._get_obs(), {"pose": pose}
```

说明：

- `super().reset(seed)` 必须调，它给 `self.np_random` 设好随机种子（`np_random.choice` 依赖它）。
- 随机挑一个倒地姿态，摆好、稳定，返回初始观测。

**运行验证**：

```powershell
python -c "from my_env import Robo1GetupEnv; e = Robo1GetupEnv('robo1.xml'); o, info = e.reset(); print(info['pose'], o)"
```

预期输出：某个姿态名（如 `roll_pos`）+ 一个 4 个浮点数的数组。

---

## Step 6：敲 `step`（核心）

继续往下敲：

```python
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        self.target += action * self.target_delta
        self.target = np.clip(self.target, -self.target_limit, self.target_limit)

        for _ in range(self.frame_skip):
            self.data.ctrl[:] = self.target
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs = self._get_obs()
```

说明（逐块）：

- 第 1 行：动作截到 `[-1,1]`，转成 float64。
- 第 3-4 行：**动作是目标角增量**——`target += action * 0.08`，再限幅 ±1.55。
- 第 6-8 行：把目标角写进 `data.ctrl`（position 执行器会推动舵机），推进 20 步仿真（=20ms）。
- 第 10-11 行：步数 +1，取新观测。

**继续往下敲**（还在 `step` 里，缩进 8 空格）——算奖励：

```python
        upright = self._upright()
        upright_progress = upright - self.prev_upright
        self.prev_upright = upright
        roll, pitch = float(obs[0]), float(obs[1])

        tilt_cost = 0.2 * (roll * roll + pitch * pitch)
        reward = 2.0 * upright + 8.0 * upright_progress - tilt_cost
```

说明：

- `upright_progress`：这一步比上一步"更直"了多少——**进步奖励**，引导它越起越高。
- `2.0 * upright`：站得越直分越高。
- `tilt_cost`：惩罚身体还在倾斜。

**继续往下敲**（`step` 结尾）——返回：

```python
        truncated = self.step_count >= self.max_steps
        return obs, float(reward), False, truncated, {"upright": upright}
```

说明：`terminated` 先恒为 `False`（核心版不判成功），`truncated` 到步数上限为真。`info` 里带出 `upright` 方便调试。

**运行验证**（跑几步随机动作）：

```powershell
python -c "from my_env import Robo1GetupEnv; e = Robo1GetupEnv('robo1.xml'); o,_ = e.reset(); [print(e.step(e.action_space.sample())[1]) for _ in range(5)]"
```

预期输出：5 个负数（随机动作起不来，被 `tilt_cost` 拉负）。**看到负数就对了**——这正是第 2 站"随机策略学不起来"的证据。

---

## Step 7：核对完整文件

到这一步，你的 `my_env.py` 应该是这样的（对照检查缩进）：

```python
import math
import numpy as np
import gymnasium as gym
import mujoco
from gymnasium import spaces


def roll_pitch_to_quat(roll, pitch):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    return np.array([cr * cp, sr * cp, cr * sp, -sr * sp], dtype=np.float64)


def quat_to_roll_pitch(q):
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    return roll, pitch


FALLEN_POSES = {
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

        self.frame_skip = 20
        self.target_delta = 0.08
        self.target_limit = 1.55
        self.settle_steps = 200
        self.max_steps = 700

        self.target = np.zeros(2, dtype=np.float64)
        self.prev_upright = 0.0
        self.step_count = 0

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
```

**正式验证**（用 SB3 自带的检查器 + 完整跑一局）：

```python
from stable_baselines3.common.env_checker import check_env
from my_env import Robo1GetupEnv

env = Robo1GetupEnv("robo1.xml")
check_env(env, warn=True)          # 通过则无报错

obs, info = env.reset()
total = 0.0
for i in range(700):
    obs, r, terminated, truncated, info = env.step(env.action_space.sample())
    total += r
    if i % 100 == 0:
        print(f"step {i}: upright={info['upright']:.3f}, reward={r:.3f}")
    if terminated or truncated:
        break
print(f"累计 reward: {total:.1f}")
```

预期：`check_env` 无报错；`upright` 一直在 0 附近（随机动作起不来）；累计 reward 为负。

---

## Step 8（进阶）：补成完整奖励

上面这个核心版已经能跑，但奖励只有三项。仓库版（`robo1_env.py:211-222`）还加了一堆惩罚项，用来**把"站起来"塑造成"干净利落地站起来"**。原理是：每个惩罚都乘一个 `stand_gate`，**只有快站直时才生效**：

```python
stand_gate = float(np.clip((upright - 0.65) / 0.35, 0.0, 1.0))
```

想把核心版补成完整版，做两件事：

1. 在 `__init__` 里补上 `servo1_qvel_id / servo2_qvel_id`（和 qpos 那两行一样，只是用 `jnt_dofadr` 查速度下标）和 `self.standing_height`（站立高度，用来算高度误差）。
2. 在 `step` 里补全奖励，以及 `goal_pose` 成功判定（`terminated` 由它决定）。

完整的 reward 和成功判定，直接对照 `robo1_env.py:179-250` 补齐即可——你已经理解了核心三条，剩下的只是"照公式加惩罚项"。

> 建议：**先用核心版把整条流程跑通**，理解了观测/动作/奖励三大件，再回来看完整版。完整版更适合直接训练时用（第 3 站）。

---

## 下一步

敲完这一站，你已经掌握了整个项目最核心的一半。接下来：

- **第 2 站**：写示范轨迹 + 行为克隆——因为你现在已经亲眼看到"随机策略 reward 为负、起不来"。
- **第 3 站**：PPO 精调——把这个 env 直接喂给 Stable-Baselines3。
