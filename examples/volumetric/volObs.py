import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append( sys.path[0] +'/../..')
from positionDMP.dmp_position import PositionDMP
from obstacles.volumetric import VolumetricObstacle
from utils.trajFuncs import generate2DTraj
from matplotlib.patches import Ellipse
import utils.plotFuncs as pf


o1 =  VolumetricObstacle(center =  np.array([-0.5,0.7]),
                        axes = np.array([.6,.4]),
                        n_dim=2,lambda_ = 10,beta = 2 , eta = 0.5)
    

dmp = PositionDMP(N_bfs=100,alpha= 50,cs_alpha=1,totaltime = 1,cs_tau = 1,n_dim = 2,obstacle = o1) ## ^ init the DMP class.

# initPos,initVel,finalPos = np.array([
#     [-0.3,0.2],
#     [0.01,-0.01],
#     [0.3,-0.2],
# ])
initPos,initVel,finalPos = np.array([
    [-3,2],
    [0,0],
    [3,-2.1],
])
# initPos,initVel,finalPos = np.array([
#     [-4,-4],
#     [-1,1],
#     [4,4],
# ])

position = generate2DTraj(initPos,initVel,finalPos,dmp.totaltime,dmp.t)
xPOS = dmp.t * np.cos(np.pi * dmp.t)
yPOS = dmp.t * np.sin(np.pi * dmp.t)

# position = np.vstack((xPOS,yPOS)).T

dmp.train(position) ## ^ train the DMP

dmp_position = dmp.rollout(position) ## ^ simulate

# o1.obstaclePos = np.array(o1.obstaclePos)

euclidiean_norm = np.linalg.norm((position - dmp_position) , axis= 1) ## ^ euclidiean norm

## ? 2D plots

fig,ax = plt.subplots(3,1,sharex= True)
ax[0].plot(dmp.t,position[:,0],label='Demo')
ax[0].plot(dmp.t,dmp_position[:,0],label='DMP')
ax[0].set_xlabel('t')
ax[0].set_ylabel('X')

ax[1].plot(dmp.t,position[:,1],label='Demo')
ax[1].plot(dmp.t,dmp_position[:,1],label='DMP')
ax[1].set_xlabel('t')
ax[1].set_ylabel('Y')
ax[1].legend()

ax[2].plot(dmp.t,euclidiean_norm,label='Error Norm')
ax[2].legend()

fig2,ax2 = plt.subplots(1,1)
ax2.plot(dmp_position[:,0],dmp_position[:,1] ,label='DMP')
ax2.plot(position[:,0],position[:,1],label='Demo')
ell = Ellipse(xy=o1.center, width=o1.axes[0], height=o1.axes[1], color='r' , fill = False)
ax2.add_artist(ell)
plt.show()
