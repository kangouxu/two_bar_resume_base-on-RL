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
    <position name="hinge_pos" joint="hinge" kp="150.0" kv="1.1"/>
  </actuator>
</mujoco>
"""

model =mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
data.ctrl[0] = 0.8          # 初始摆角 0.8 rad（约 45°）

for i in range(10000):   #运行10000个步长
    mujoco.mj_step(model, data)  #步进
    if i % 100 == 0:  #每200步输出一次数据
        print(f"t={i * 0.001:.1f}s   目标角={data.ctrl[0]:.3f}  实际角={data.qpos[0]:.3f}")