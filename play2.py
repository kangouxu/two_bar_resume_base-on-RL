import mujoco

XML = """
<mujoco>
  <option timestep="0.001"/>           #仿真步长
  <worldbody>
    <body name="pole" pos="0 0 1">  #初始位置
      <joint name="hinge" type="hinge" axis="1 0 0" damping="0.1"/>   #关节类型
      <geom type="box" size="0.02 0.02 0.5" pos="0 0 -0.5"/>   #体积参数
    </body>
    <geom name="floor" type="plane" size="2 2 0.1"/>   #地面  平坦  大小2*2
  </worldbody>
</mujoco>
"""

model =mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
data.qpos[0] = 0.8          # 初始摆角 0.8 rad（约 45°）

for i in range(1000):   #运行1000个步长
    mujoco.mj_step(model, data)  #步进
    if i % 100 == 0:  #每200步输出一次数据
        print(f"t={i * 0.001:.1f}s   摆角={data.qpos[0]:.4f}")