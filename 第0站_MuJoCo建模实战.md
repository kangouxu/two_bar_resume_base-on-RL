# 第 0 站：从零敲出 MuJoCo 模型

> 这一站不碰强化学习，只做一件事：**一行一行敲出几个 MuJoCo 模型，搞懂 MJCF 建模**。
> 和第 1 站不同，这一站是几个**各自独立的小实验**（每个都能单独运行），而不是一个文件长成。
> 敲完你就能读懂 `robo1.xml` 的每一行。
>
> 对照参考：`RL/robo1.xml`。

---

## Step 0：确认 MuJoCo 装好

```powershell
python -c "import mujoco; print(mujoco.__version__)"
```

预期输出：一个版本号（如 `3.2.x`）。报错就 `pip install mujoco`。

---

## Step 1：敲第一个模型——自由落体的方块

新建 `play1.py`，敲入：

```python
import mujoco

XML = """
<mujoco>
  <option timestep="0.001"/>
  <worldbody>
    <body name="box" pos="0 0 0.5">
      <joint type="free"/>
      <geom type="box" size="0.05 0.05 0.05"/>
    </body>
    <geom name="floor" type="plane" size="2 2 0.1"/>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

for i in range(1000):
    mujoco.mj_step(model, data)
    if i % 200 == 0:
        print(f"t={i * 0.001:.1f}s   z={data.qpos[2]:.4f}")
```

逐段说：

- `<option timestep="0.001"/>`：仿真步长 1ms。
- `<body name="box" pos="0 0 0.5">`：一个刚体，初始悬空 0.5m。
- `<joint type="free"/>`：给这个刚体 **6 个自由度**（能自由移动 + 旋转），所以它能自由落体。
- `<geom type="box" size="0.05 0.05 0.05"/>`：它的形状是一个方块（`size` 是**半**尺寸）。
- `<geom type="plane">`：地面，静态的（不挂在 body 下）。
- `from_xml_string`：直接用字符串建模型（不用建 .xml 文件）。
- `MjData(model)`：建运行状态 `data`。之后推进仿真都是在动 `data`。

**运行验证**：

```powershell
python play1.py
```

预期：`z` 从 `0.5000` 一路降到 `0.0500` 附近（方块落到地面，方块半高 0.05，所以中心停在 z=0.05）。

> 这就是物理引擎在算重力 + 碰撞。MuJoCo 会自动从 geom 尺寸推断质量和惯性，所以不用写 `inertial`。

---

## Step 2：改成单摆——加铰链

新建 `play2.py`（在 Step 1 基础上改 XML），敲入：

```python
import mujoco

XML = """
<mujoco>
  <option timestep="0.001"/>
  <worldbody>
    <body name="pole" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="1 0 0" damping="0.1"/>
      <geom type="box" size="0.02 0.02 0.5" pos="0 0 -0.5"/>
    </body>
    <geom name="floor" type="plane" size="2 2 0.1"/>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
data.qpos[0] = 0.8          # 初始摆角 0.8 rad（约 45°）

for i in range(1000):
    mujoco.mj_step(model, data)
    if i % 100 == 0:
        print(f"t={i * 0.001:.1f}s   摆角={data.qpos[0]:.4f}")
```

相比 Step 1，改了三处：

- `free` 换成 `hinge`：**1 个自由度**的旋转。所以 `qpos` 现在只占 1 位（角度），用 `data.qpos[0]` 读。
- `axis="1 0 0"`：转轴是 x 轴（绕哪个轴转）。
- `geom` 的 `pos="0 0 -0.5"`：把杆的重心偏移到关节下方，重力会让它竖直垂下——这就是单摆。

`damping="0.1"` 是阻尼，让摆慢慢停下来（不写的话会一直摆）。

**运行验证**：

```powershell
python play2.py
```

预期：摆角从 `0.8000` 逐渐衰减到接近 `0`（竖直）。

> **这就是 `robo1.xml` 里 `servo1_joint`（绕 x）和 `servo2_joint`（绕 y）的本质**——两个正交的 hinge 关节。

---

## Step 3：加执行器——从"能动"到"能控制"

新建 `play3.py`，在 Step 2 基础上加 `<actuator>`，并去掉 damping：

```python
import mujoco

XML = """
<mujoco>
  <option timestep="0.001"/>
  <worldbody>
    <body name="pole" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="1 0 0"/>
      <geom type="box" size="0.02 0.02 0.5" pos="0 0 -0.5"/>
    </body>
    <geom name="floor" type="plane" size="2 2 0.1"/>
  </worldbody>
  <actuator>
    <position name="hinge_pos" joint="hinge" kp="2.0" kv="0.2"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
data.ctrl[0] = 0.8          # 目标角 0.8 rad

for i in range(500):
    mujoco.mj_step(model, data)
    if i % 100 == 0:
        print(f"t={i * 0.001:.1f}s   目标角={data.ctrl[0]:.3f}  实际角={data.qpos[0]:.3f}")
```

新增的 `<actuator>` 说明：

- `<position ... joint="hinge" ...>`：一个 **position 执行器**，驱动 `hinge` 关节——**位置控制**，把它"推向"目标角。
- `kp="2.0"`（比例增益）：追得多快/多硬。
- `kv="0.2"`（阻尼）：稳不稳。

`data.ctrl[0] = 0.8` 就是给这个执行器下命令"转到 0.8 rad"。

**运行验证**：

```powershell
python play3.py
```

预期：`实际角` 快速逼近 `目标角`（0.8），先冲过头一点再稳定下来（kp/kv 决定这个动态）。

> **这一步是整个项目"仿真↔实机"对齐的根基**：真实舵机也是位置控制，所以用 `position` 执行器模拟它，动作语义才一致。`robo1.xml:17-19` 里的 `kp="0.5" kv="0.04"` 就是这俩参数。

> 三种执行器对比：`position`（控角度，像舵机）、`velocity`（控转速，像电机）、`motor`（控力矩）。本项目用 `position`。

---

## Step 4：逐段拆解 `robo1.xml`

现在你懂了 body / joint / geom / actuator，打开 `RL/robo1.xml` 逐段读（不用敲，读就行）：

**① 头部**（`robo1.xml:2-4`）

```xml
<compiler angle="radian" meshdir="assets" autolimits="true"/>
<option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast"/>
```

- `angle="radian"`：所有角度用**弧度**（所以 ±90° 写成 ±1.57079632679）。
- `meshdir="assets"`：网格文件从 `assets/` 目录找。
- `timestep="0.001"`：**这行决定了后面 `frame_skip=20` = 20ms = 50Hz 的换算**。

**② 资源**（`robo1.xml:6-13`）：`<mesh name="servo1" file="servo1.stl"/>` 等——把 STL 网格加载进来命名，给 geom 用。这些 STL 是 3D 打印件的数字孪生。

**③ 默认参数**（`robo1.xml:15-20`）

```xml
<default>
  <joint type="hinge" limited="true" range="-1.57079632679 1.57079632679"/>
  <position ctrllimited="true" ctrlrange="-1.57079632679 1.57079632679"
            forcelimited="true" forcerange="-1.0 1.0" kp="0.5" kv="0.04"/>
</default>
```

`<default>` 里的参数**自动套用到所有同类元素**（除非被覆盖）——这样每个 joint 不用重复写 range。

**④ 主体 body 树**（`robo1.xml:22-86`）：反映真实机械结构：

```
foot（脚，free joint，能自由落地）
├── ATOM（M5Atom 单片机）
├── servo1（舵机 1）
└── arm1（手臂 1，servo1_joint 绕 x 轴）
    └── servo2（舵机 2）
        └── arm2（手臂 2，servo2_joint 绕 y 轴）
```

**body 嵌套 = 运动学链**。`arm1` 挂在 `foot` 下、通过 `servo1_joint` 绕 x 转；`arm2` 挂在 `arm1` 下、绕 y 转。两个正交铰链就是起身的全部自由度。

**⑤ 执行器**（`robo1.xml:88-91`）

```xml
<position name="servo1_pos" joint="servo1_joint"/>
<position name="servo2_pos" joint="servo2_joint"/>
```

执行器**顺序 = `data.ctrl` 的顺序**：`data.ctrl[0]` 控 servo1，`data.ctrl[1]` 控 servo2。

---

## Step 5：综合练习——自己写一个 .xml 文件

把 Step 3 的 XML **单独存成 `pendulum.xml`**，再写 `play5.py` 用 `from_xml_path` 加载，用正弦波驱动它：

```python
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("pendulum.xml")
data = mujoco.MjData(model)

for i in range(2000):
    data.ctrl[0] = 0.5 * np.sin(i * 0.01)     # 目标角随时间正弦变化
    mujoco.mj_step(model, data)
    if i % 200 == 0:
        print(f"t={i * 0.001:.1f}s   目标={data.ctrl[0]:.3f}  实际={data.qpos[0]:.3f}")
```

**运行验证**：`python play5.py`，预期：`实际` 追着 `目标` 走，但有延迟和误差（kp 越大追得越紧）。

> 观察"目标角"和"实际角"的关系——这就是第 1 站 env 里 `data.ctrl[:] = self.target` 那行的物理含义。

---

## 概念速查表

| 概念 | 说明 |
| --- | --- |
| `data.qpos` | 所有关节位置（free=7 位，hinge=1 位） |
| `data.qvel` | 所有关节速度 |
| `data.ctrl` | 执行器命令（position 执行器 = 目标角） |
| free joint | 6 自由度，身体能自由落地 |
| hinge joint | 1 自由度旋转，`axis` 指定转轴 |
| `limited`/`range` | 是否限位 + 角度范围 |
| position 执行器 | 位置控制（PD 闭环），`kp`/`kv` 调响应 |

---

## 下一步

敲完这一站，你应该能：① 读懂 `robo1.xml` 每一行；② 自己写一个简单的 MJCF 模型并控制它。

下一步进入 **第 1 站**：在 `robo1.xml` 之上，用 gymnasium 接口敲出观测、动作、奖励。
