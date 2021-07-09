import pybullet as pb
import pybullet_data
import numpy as np

pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
kuka = pb.loadURDF("kuka_iiwa/model.urdf",[0,0,0],useFixedBase=True)
plane = pb.loadURDF("plane.urdf")


while pb.isConnected():
    pass
    