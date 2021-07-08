import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append( sys.path[0] +'/../..')
from positionDMP.dmp_position import PositionDMP
from time import sleep

import pybullet as pb
import pybullet_data

pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())


kuka = pb.loadURDF("kuka_iiwa/model.urdf",[0,0,0],useFixedBase=True)

plane = pb.loadURDF("plane.urdf")

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
controlJoints = []

totalJoints = pb.getNumJoints(kuka)
for i in range(totalJoints):
    jointInfo = pb.getJointInfo(kuka,i)
            
    if jointInfo[2]==0:
        #append the joints we can control
        controlJoints.append(i)
print("control",controlJoints)
i = 0 ## init the counter.
for i in range(len(dmp_position)):
    
    pb.setJointMotorControlArray(kuka,controlJoints,pb.POSITION_CONTROL,dmp_position[i])
    pb.stepSimulation()
    sleep(0.01)
    i+=1
    pass
