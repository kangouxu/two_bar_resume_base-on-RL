import mujoco
model =mujoco.MjModel.from_xml_path("play4.xml")
data = mujoco.MjData(model)
data.ctrl[0] = 0.8          # 初始摆角 0.8 rad（约 45°）

for i in range(10000):   #运行10000个步长
    mujoco.mj_step(model, data)  #步进
    if i % 200 == 0:  #每200步输出一次数据
        print(f"t={i * 0.001:.1f}s   目标角={data.ctrl[0]:.3f}  实际角={data.qpos[0]:.3f}")