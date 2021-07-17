import pybullet as pb
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append( sys.path[0] +'/../..')
from positionDMP.dmp_position import PositionDMP
from positionDMP.dmp_quat import QuaternionDMP
import utils.trajFuncs as tf
from utils.manipulation.manipulator import Manipulator
from time import sleep
import quaternionic as qt

totaltime = 10
#kuka.setInitPos(endEffectorPos= [0.3 0.2 0.5] , endEffectorOrientation= [0,0,0,1])

## init the DMP class for both position and quaternion

dmp = PositionDMP(N_bfs=100,alpha= 20,cs_alpha=0.5,totaltime = totaltime,cs_tau = 1) ## ^ init the DMP class.
quatDMP = QuaternionDMP(alpha = 40, cs_alpha = 1,N_bfs=100,totaltime = totaltime)  ## ^ init the DMP quaternion class

## generate position and orientation trajectory
initPos,initVel,finalPos = np.array([
    [0,0,1],
    [0,0,0],
    [0.1,0.2,0.8],
])
quatDMP.initQuat = qt.array([1,0.1,0.2,0]).normalized
quatDMP.goalQuat = qt.array([0,1,0,1]).normalized

## init pybullet env
kuka = Manipulator(initEndeffectorPos= initPos,initEndeffectorOrientation=
                   tf.convertTo_pybulletQuat(quatDMP.initQuat))

## generate the demo trajectories.
position = tf.generate3DTraj(initPos,initVel,finalPos,dmp.totaltime,dmp.t)
rotation = tf.generateQuaternionTraj(quatDMP.initQuat,quatDMP.goalQuat,quatDMP.t/quatDMP.totaltime)

## train the DMP.
dmp.train(position) ## ^ train the DMP
quatDMP.train(rotation)
dmp_position = dmp.rollout(position) ## ^ simulate
dmp_quaternion = quatDMP.rollout(rotation)
dmp_pbQuat = tf.convertTo_pybulletQuat(dmp_quaternion)

# sanity check
if len(dmp_position) != len(dmp_pbQuat):
    RuntimeError("len of dmp position and quaternion are not same")

pb_orient,pb_position = [],[]

for i in range(len(dmp.t)):
    
    kuka.setParams()
    kuka.getInverseKinematics(dmp_position[i],dmp_pbQuat[i])
    #kuka.getInverseKinematics(dmp_position[i])
    pb.setJointMotorControlArray(kuka.armID,kuka.controlJoints,pb.POSITION_CONTROL,
                                 targetPositions = kuka.kinematics.inv_jointPosition)
    
    pb_orient.append(kuka.kinematics.linkOrientation)
    pb_position.append(kuka.kinematics.linkPosition)
    pb.stepSimulation()
    sleep(0.01)

pb_orient = np.array(pb_orient)
pb_position = np.array(pb_position)

## ? 2D plots.
fig,ax = plt.subplots(4,1,sharex= True)

ax[0].plot(dmp.t,position[:,0],label='Demo')
ax[0].plot(dmp.t,dmp_position[:,0],label='DMP')
ax[0].plot(dmp.t,pb_position[:,0],label='pybullet')
ax[0].set_xlabel('t')
ax[0].set_ylabel('X')

ax[1].plot(dmp.t,position[:,1],label='Demo')
ax[1].plot(dmp.t,dmp_position[:,1],label='DMP')
ax[1].plot(dmp.t,pb_position[:,1],label='pybullet')
ax[1].set_xlabel('t')
ax[1].set_ylabel('Y')

ax[2].plot(dmp.t,position[:,2],label='Demo')
ax[2].plot(dmp.t,dmp_position[:,2],label='DMP')
ax[2].plot(dmp.t,pb_position[:,2],label='pybullet')
ax[2].set_xlabel('t')
ax[2].set_ylabel('z')
ax[2].legend()

euclidiean_norm = np.linalg.norm((position - dmp_position) , axis= 1) ## ^ euclidiean norm for position.
ax[3].plot(dmp.t,euclidiean_norm,label='Error Norm')
ax[3].legend()

rotation = rotation.ndarray
dmp_quaternion = dmp_quaternion.ndarray

rotationError = np.linalg.norm(rotation - dmp_quaternion,axis=1)

figQuat,axQuat = plt.subplots(5,1,sharex=True)

axQuat[0].plot(quatDMP.t,rotation[:,0],label='Demo')
axQuat[0].plot(quatDMP.t,dmp_quaternion[:,0],label = "DMP")
axQuat[0].plot(quatDMP.t,pb_orient[:,3],label = "pybullet")
axQuat[0].set_xlabel('T(s)')
axQuat[0].set_ylabel('W')

axQuat[1].plot(quatDMP.t,rotation[:,1],label='Demo')
axQuat[1].plot(quatDMP.t,dmp_quaternion[:,1],label='Demo')
axQuat[1].plot(quatDMP.t,pb_orient[:,0],label = "pybullet")
axQuat[1].set_ylabel('X')

axQuat[2].plot(quatDMP.t,rotation[:,2],label='Demo')
axQuat[2].plot(quatDMP.t,dmp_quaternion[:,2],label='DMP')
axQuat[2].plot(quatDMP.t,pb_orient[:,1],label = "pybullet")
axQuat[2].set_ylabel('Y')

axQuat[3].plot(quatDMP.t,rotation[:,3],label='Demo')
axQuat[3].plot(quatDMP.t,dmp_quaternion[:,3],label = "DMP")
axQuat[3].plot(quatDMP.t,pb_orient[:,2],label = "pybullet")
axQuat[3].set_ylabel('Z')
axQuat[3].legend()

axQuat[4].plot(quatDMP.t,rotationError,label='error')
axQuat[4].legend()


## ? 3D plot.

fig3D = plt.figure()
ax3D = plt.axes(projection = '3d')
ax3D.plot3D(position[:,0],position[:,1],position[:,2],label='Demo')
ax3D.plot3D(dmp_position[:,0],dmp_position[:,1],dmp_position[:,2],label='DMP')
ax3D.set_xlabel('X')
ax3D.set_ylabel('Y')
ax3D.set_zlabel('Z')
ax3D.legend()

plt.show()

    