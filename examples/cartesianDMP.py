import numpy as np
import matplotlib.pyplot as plt
plt.style.use('science')
import sys
sys.path.append( sys.path[0] +'/..')
from positionDMP.dmp_position import PositionDMP
from utils.trajFuncs import generate3DTraj

dmp = PositionDMP(N_bfs=100,alpha= 30,cs_alpha=3,totaltime = 5,cs_tau = 1) ## ^ init the DMP class.

initPos,initVel,finalPos = np.array([
    [-3,-1,1],
    [0,0,-2],
    [2,3,2],
])
#position = np.array([np.sin(dmp.t),np.cos(dmp.t),np.sin(dmp.t) * np.cos(dmp.t)]).T ## ^ set the desired position.

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
ax[2].set_ylabel('Z')
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
