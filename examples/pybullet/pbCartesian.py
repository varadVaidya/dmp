import pybullet as pb
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append( sys.path[0] +'/../..')
from positionDMP.dmp_position import PositionDMP
from utils.trajFuncs import generate3DTraj
from utils.manipulation.manipulator import Manipulator
from time import sleep
## init pybullet env

kuka = Manipulator()
dmp = PositionDMP(N_bfs=100,alpha= 20,cs_alpha=0.5,totaltime = 10,cs_tau = 1) ## ^ init the DMP class.
## generate position
initPos,initVel,finalPos = np.array([
    [0.2,0.2,0.3],
    [0,0,0],
    [0.1,-0.2,0.5],
])

position = generate3DTraj(initPos,initVel,finalPos,dmp.totaltime,dmp.t)
dmp.train(position) ## ^ train the DMP

dmp_position = dmp.rollout(position) ## ^ simulate

euclidiean_norm = np.linalg.norm((position - dmp_position) , axis= 1) ## ^ euclidiean norm


## ? 2D plots.
fig,ax = plt.subplots(4,1,sharex= True)

ax[0].plot(dmp.t,position[:,0],label='Demo')
ax[0].plot(dmp.t,dmp_position[:,0],label='DMP')
ax[0].set_xlabel('t')
ax[0].set_ylabel('X')

ax[1].plot(dmp.t,position[:,1],label='Demo')
ax[1].plot(dmp.t,dmp_position[:,1],label='DMP')
ax[1].set_xlabel('t')
ax[1].set_ylabel('Y')

ax[2].plot(dmp.t,position[:,2],label='Demo')
ax[2].plot(dmp.t,dmp_position[:,2],label='DMP')
ax[2].set_xlabel('t')
ax[2].set_ylabel('z')
ax[2].legend()

ax[3].plot(dmp.t,euclidiean_norm,label='Error Norm')
ax[3].legend()

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
for i in range(len(dmp.t)):
    
    kuka.getInverseKinematics(dmp_position[i])
    pb.setJointMotorControlArray(kuka.armID,kuka.controlJoints,pb.POSITION_CONTROL,targetPositions = kuka.kinematics.inv_jointPosition)    
    pb.stepSimulation()
    sleep(0.01)
    