import pybullet as pb
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append( sys.path[0] +'/../..')
from positionDMP.dmp_position import PositionDMP
from positionDMP.dmp_quat import QuaternionDMP
import utils.trajFuncs as tf
import utils.plotFuncs as pf
from utils.manipulation.manipulator import Manipulator
from time import sleep
import quaternionic as qt
import pybullet_utils.transformations as trans

kuka = Manipulator(basePosition=[0,0.5,1.25],baseOrientation=pb.getQuaternionFromEuler((np.pi/2,0,0)))
pb.loadURDF("table/table.urdf", basePosition=[0,0,0],useFixedBase= True,baseOrientation=pb.getQuaternionFromEuler((0,0,0)) )

pb.setJointMotorControlArray(kuka.armID ,kuka.controlJoints ,pb.VELOCITY_CONTROL,forces = kuka.controlZero)

while pb.isConnected():
    kuka.setParams()
    print(kuka.kinematics.linkPosition)
    pb.stepSimulation()
    sleep(0.01)