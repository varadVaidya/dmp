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

totaltime = 10
#kuka.setInitPos(endEffectorPos= [0.3 0.2 0.5] , endEffectorOrientation= [0,0,0,1])

## init the DMP class for both position and quaternion

dmp = PositionDMP(N_bfs=100,alpha= 20,cs_alpha=0.5,totaltime = totaltime,cs_tau = 1) ## ^ init the DMP class.
quatDMP = QuaternionDMP(alpha = 40, cs_alpha = 1,N_bfs=100,totaltime = totaltime)  ## ^ init the DMP quaternion class

## generate position and orientation trajectory
initPos,initVel,finalPos = np.array([
    [0,0,1],
    [0,0,0],
    [-0.1,0.2,0.8],
])
quatDMP.initQuat = qt.array([1,0.1,0.2,0]).normalized
quatDMP.goalQuat = qt.array([-0.5,1,0.5,1]).normalized

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

Kp = 550 * np.diag([0.1,0.1,0.1,0.08,0.08,0.08])
def velocityControl(i):
    
    positionError = dmp_position[i] - np.array(kuka.kinematics.linkPosition)   
    currentQuat = kuka.kinematics.linkOrientation
    desiredOrientation = dmp_pbQuat[i]
    
    errorQuat = trans.quaternion_multiply(desiredOrientation, 
                                          trans.quaternion_conjugate(currentQuat))
    
    orientationError = errorQuat[0:3] * np.sign(errorQuat[3])
    
    posError = np.hstack((positionError,orientationError))
        
    commandVelocity = Kp.dot(posError)
    commandJointVelocity = kuka.kinematics.geometricJacobianInv.dot(commandVelocity)
    
    pb.setJointMotorControlArray(kuka.armID ,kuka.controlJoints ,pb.VELOCITY_CONTROL,targetVelocities=commandJointVelocity)    
    

for i in range(len(dmp.t)):
    
    kuka.setParams()
    
    
    # kuka.getInverseKinematics(dmp_position[i],dmp_pbQuat[i])
    # #kuka.getInverseKinematics(dmp_position[i])
    # pb.setJointMotorControlArray(kuka.armID,kuka.controlJoints,pb.POSITION_CONTROL,
    #                              targetPositions = kuka.kinematics.inv_jointPosition)
    velocityControl(i)
    pb_orient.append(kuka.kinematics.linkOrientation)
    pb_position.append(kuka.kinematics.linkPosition)
    pb.stepSimulation()
    sleep(0.01)

pb_orient = np.array(pb_orient)
pb_position = np.array(pb_position)
pf.plotPosition(dmp.t,position,dmp_position,pb_position)
pf.plotQuaternions(quatDMP.t,rotation.ndarray,dmp_quaternion.ndarray,pb_orient) ## pass the md array version of the quaternion class matrix

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

    