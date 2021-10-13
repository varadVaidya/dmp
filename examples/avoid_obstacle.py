import numpy as np
import matplotlib.pyplot as plt
plt.style.use('science')
import sys
sys.path.append( sys.path[0] +'/..')
from positionDMP.dmp_position import PositionDMP
from obstacles.obstacle import Obstacle
from utils.trajFuncs import generate3DTraj
import utils.plotFuncs as pf

## init the obstacle
extrapolteFlag = False
o1 = Obstacle(initPos=np.array([0,0,0]) ,lambda_ = 5)
dmp = PositionDMP(N_bfs=100,alpha= 30,cs_alpha=5,totaltime = 5,obstacle = o1,extrapolate= extrapolteFlag) ## ^ init the DMP class.

initPos,initVel,finalPos = np.array([
    [3,2,3],
    [-1,0,-1.3],
    [-4,-5,-3],
])

position = generate3DTraj(initPos,initVel,finalPos,dmp.totaltime,dmp.t)
dmp.train(position) ## ^ train the DMP

dmp_position = dmp.rollout(position) ## ^ simulate
o1.obstaclePos = np.array(o1.obstaclePos)
euclidiean_norm = np.linalg.norm((position - dmp_position) , axis= 1) ## ^ euclidiean norm

## plot stuff...
pf.plotPosition(dmp.t,position,dmp_position)
# pf.animatePositionDMP3D(dmp.t,position,dmp_position,o1.obstaclePos)

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




    