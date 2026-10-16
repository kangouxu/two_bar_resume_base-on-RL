import mujoco

XML = """
<mujoco>
  <option timestep="0.001"/>           #仿真步长
  <worldbody>
    <body name="box" pos="0 0 0.9">  #初始位置
      <joint type="free"/>    #关节类型
      <geom type="box" size="0.05 0.05 0.05"/>   #体积参数
    </body>
    <geom name="floor" type="plane" size="2 2 0.1"/>   #地面  平坦  大小2*2
  </worldbody>
</mujoco>
"""

model =mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

for i in range(1000):   #运行1000个步长
    mujoco.mj_step(model, data)  #步进
    if i % 200 == 0:  #每200步输出一次数据
        print(f"t={i * 0.001:.1f}s   z={data.qpos[2]:.4f}")