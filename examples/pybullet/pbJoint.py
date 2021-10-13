import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append( sys.path[0] +'/../..')
from positionDMP.dmp_position import PositionDMP
from utils.manipulation.manipulator import Manipulator

from time import sleep

import pybullet as pb
import pybullet_data



dmp = PositionDMP(N_bfs=1000,alpha= 10,cs_alpha=0.5,totaltime = 15,n_dim=7,cs_tau = 1) ## ^ init the DMP class.
# position = np.array([np.sin(dmp.t),np.cos(dmp.t),np.sin(dmp.t) * np.cos(dmp.t)]).T ## ^ set the desired position.

position = [None] * dmp.n_dim

for i in range(dmp.n_dim):
    position[i] = np.sin(dmp.t)

position = np.array(position).T
dmp.train(position) ## ^ train the DMP

dmp_position = dmp.rollout(position) ## ^ simulate
euclidiean_norm = np.linalg.norm((position - dmp_position) , axis= 1) ## ^ euclidiean norm
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


plt.show()

kuka = Manipulator()
sleep(10)
i = 0 ## init the counter.
smolBallScale = [0.009,0.009,0.009]
smolBall = pb.createVisualShape(shapeType=pb.GEOM_MESH,
                                    fileName="sphere_smooth.obj",
                                    meshScale=smolBallScale,
                                    rgbaColor=[1,0,0,1])
for i in range(len(dmp_position)):
    
    kuka.setParams()
    pb.setJointMotorControlArray(kuka.armID,kuka.controlJoints,pb.POSITION_CONTROL,dmp_position[i])
    if i % 100 == 0:
        pb.createMultiBody(baseMass=0,
                        baseInertialFramePosition=[0, 0, 0],
                    baseVisualShapeIndex=smolBall,
                    basePosition=np.array(kuka.kinematics.linkPosition))

    pb.stepSimulation()
    
    pass
