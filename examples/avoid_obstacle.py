import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append( sys.path[0] +'/..')
from positionDMP.dmp_position import PositionDMP
from obstacles.obstacle import Obstacle
from utils.trajFuncs import generate3DTraj

## init the obstacle
o1 = Obstacle(initPos=np.array([-0.3,-0.4,-0.3]))
dmp = PositionDMP(N_bfs=100,alpha= 30,cs_alpha=5,totaltime = 5,obstacle = o1) ## ^ init the DMP class.

initPos,initVel,finalPos = np.array([
    [-2,-2,-2],
    [0,0,0],
    [1,1,1.1]
])

position = generate3DTraj(initPos,initVel,finalPos,dmp.totaltime,dmp.t)
dmp.train(position) ## ^ train the DMP

dmp_position = dmp.rollout(position) ## ^ simulate

euclidiean_norm = np.linalg.norm((position - dmp_position) , axis= 1) ## ^ euclidiean norm

## plot stuff...

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





## ^ uncommented the below code as the below is for 3D obstacle.
## ? 3D plot.

fig3D = plt.figure()
ax3D = plt.axes(projection = '3d')
ax3D.plot3D(position[:,0],position[:,1],position[:,2],label='Demo')
ax3D.plot3D(dmp_position[:,0],dmp_position[:,1],dmp_position[:,2],label='DMP')
ax3D.plot([o1.initPos[0]],[o1.initPos[1]],[o1.initPos[2]], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5)
ax3D.set_xlabel('X')
ax3D.set_ylabel('Y')
ax3D.set_zlabel('Z')
ax3D.legend()

plt.show()

    