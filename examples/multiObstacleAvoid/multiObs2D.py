import numpy as np
import matplotlib.pyplot as plt
plt.style.use('science')
import sys
sys.path.append( sys.path[0] +'/../..')
from positionDMP.dmp_position import PositionDMP
from obstacles.multiObstacle import Obstacle
from utils.trajFuncs import generate2DTraj
import utils.plotFuncs as pf

obsPos = np.array([
            [1,1.8],
            [-2,0.2],
            [0,1]
        ])

o1 = Obstacle(numObs = 3 , n_dim = 2 , initPos = obsPos,lambda_ = 10)

dmp = PositionDMP(N_bfs=100,alpha= 30,cs_alpha=3,totaltime = 5,cs_tau = 1,n_dim = 2,obstacle = o1) ## ^ init the DMP class.

initPos,initVel,finalPos = np.array([
    [-4,-4],
    [-1,1],
    [4,4],
])

position = generate2DTraj(initPos,initVel,finalPos,dmp.totaltime,dmp.t)

dmp.train(position) ## ^ train the DMP

dmp_position = dmp.rollout(position) ## ^ simulate

o1.obstaclePos = np.array(o1.obstaclePos)

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
ax2.plot(position[:,0],position[:,1],label='demo')
ax2.plot(dmp_position[:,0],dmp_position[:,1],label='dmp')
ax2.plot(o1.initPos[0,0],o1.initPos[0,1],'ro',label='obstacle')
ax2.plot(o1.initPos[1,0],o1.initPos[1,1],'ro')
ax2.plot(o1.initPos[2,0],o1.initPos[2,1],'ro')

ax2.legend()
plt.show()

# pf.animatePositionDMP2D(dmp.t,position,dmp_position,obstaclePosition= o1.obstaclePos,saveVideo=True)   

