# 第 3 站：从零敲出 PPO 精调

> 第 2 站得到一个"能大致起身"但只会死记硬背的初始策略。
> 这一站**敲出 `train_robo1.py`**，用 PPO 在仿真里继续精调，让它变成"真会应对各种情况"的策略。
>
> 对照实现：`RL/train_robo1.py`、`RL/eval_robo1_policy.py`、`RL/play_robo1_policy.py`。

---

## 先花 1 分钟搞懂 PPO（概念）

PPO 不需要手写，但懂三个词就能理解它在干嘛：

1. **Actor-Critic 两个网络**：Actor（策略网络）负责"看状态、出动作"——就是第 2 站克隆的那个；Critic（价值网络）负责估"这个状态能拿多少长期奖励"，用来判断"这步动作比预期好还是坏"。
2. **Advantage（优势）**：实际拿到的奖励比 Critic 的预期高，就说明这个动作好，下次多做。
3. **Clip**：限制策略每轮更新别跨太大步，保证训练稳。

一句话：**Actor 做，Critic 评，PPO 让 Actor 每次只改进一点点。**

---

## Step 0：确认有预训练模型

第 2 站你存了 `robo1_getup_ppo.zip`（或仓库自带一个训练好的）。确认：

```powershell
python -c "from stable_baselines3 import PPO; PPO.load('robo1_getup_ppo'); print('ok')"
```

预期输出：`ok`

---

## Step 1：敲 train_robo1.py

新建 `train_robo1.py`，敲：

```python
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from robo1_env import Robo1GetupEnv


def make_env():
    def _init():
        return Robo1GetupEnv("robo1.xml")
    return _init
```

说明：

- `SubprocVecEnv`：用多进程开多个环境并行 roll-out，加速训练。
- `make_env`：返回一个**工厂函数**，SubprocVecEnv 每个进程用它各建一个环境。

继续往下敲 `main`：

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-in", default=None, help="从已有模型继续训练")
    parser.add_argument("--model-out", default="robo1_getup_ppo", help="输出模型名")
    args = parser.parse_args()

    check_env(Robo1GetupEnv("robo1.xml"), warn=True)
    env = SubprocVecEnv([make_env() for _ in range(args.n_envs)])
    env.seed(args.seed)

    if args.model_in:
        model = PPO.load(args.model_in, env=env, device="cpu")
    else:
        model = PPO("MlpPolicy", env, verbose=1, n_steps=512, batch_size=256,
                    learning_rate=3e-4, gamma=0.99, seed=args.seed, device="cpu")

    model.learn(total_timesteps=args.timesteps)
    model.save(args.model_out)
    env.close()


if __name__ == "__main__":
    main()
```

说明（关键几行）：

- `check_env`：训练前先检查环境接口规范，有问题早发现。
- `PPO.load(...)`：**从预训练继续**（这就是整个项目"先克隆、再精调"的衔接点）。
- `PPO("MlpPolicy", ...)`：**从零训练**（省略 `--model-in` 时走这条）。
- 超参数：`n_steps=512`（每轮收多少步经验）、`batch_size=256`、`learning_rate=3e-4`、`gamma=0.99`（未来奖励多重要，近 1=重视长期）。

---

## Step 2：运行训练，观察日志

从预训练继续：

```powershell
python train_robo1.py --model-in robo1_getup_ppo --timesteps 200000 --n-envs 6 --model-out my_ppo
```

预期：SB3 打印 rollout 日志，其中 `ep_rew_mean`（平均单局奖励）应该**在正数区间，并随训练稳定/上升**。如果一直是负的，说明预训练起点就错了。

> 如果你没装多核或内存小，把 `--n-envs` 降到 2。CPU 训练，20 万步一般几分钟到十几分钟。

---

## Step 3：敲完整的评估脚本

第 2 站你敲过简版 `my_eval.py`（只看 `best_upright`）。这里补一个**看 `goal_pose`** 的完整版。新建 `my_eval.py`，改成：

```python
from stable_baselines3 import PPO
from robo1_env import Robo1GetupEnv

model = PPO.load("my_ppo")          # 换成你训练的模型名

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
```

说明：

- `deterministic=True`：评估时用确定性动作（不做探索）。
- `goal_pose`：严格成功标准（站直 + 身体稳 + 手臂回正），比 `best_upright` 更硬。

**运行验证**：

```powershell
python my_eval.py
```

预期：四种姿态 `best_upright` 接近 1，`goal_pose` 全为 `True`。

---

## Step 4：回放——亲眼看它起身

评估只给数字，想"看"它怎么起，用仓库现成的回放脚本（不用敲，直接跑）：

```powershell
python play_robo1_policy.py --pose pitch_pos
```

> 提示：`play_robo1_policy.py` 里模型路径是写死的（第 18 行 `PPO.load("robo1_getup_ppo")`，且没有 `--model` 参数）。如果你训练输出叫 `my_ppo`，先把那一行改成 `PPO.load("my_ppo")` 再跑。

---

## Step 5：必做的对比实验

为了真正理解"预训练"的价值，做这个对比：

```powershell
# 实验 A：从预训练继续（正常流程）
python train_robo1.py --model-in robo1_getup_ppo --timesteps 200000 --model-out model_A

# 实验 B：从零训练
python train_robo1.py --timesteps 200000 --model-out model_B
```

分别把 `my_eval.py` 里的模型路径改成 `model_A`、`model_B` 各跑一次评估。

你大概率会发现：**model_A 很快稳定起身，model_B 20 万步也起不来**。这个差距就是"稀疏奖励难探索"最直观的证据——比任何文字都有说服力。

---

## 下一步

到这一步，你在**仿真里**有了一个鲁棒的起身策略。最后一步（**第 4 站**）把它从 Python 搬到 M5Atom 实机。

而这一步能成，是因为第 1 站就埋好了三条"对齐"：观测、动作、频率——它们在第 4 站逐一兑现。
