import pybullet as pb
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append( sys.path[0] +'/../..')
from positionDMP.dmp_position import PositionDMP
from positionDMP.dmp_quat import QuaternionDMP
from obstacles.multiObstacle import Obstacle
import utils.trajFuncs as tf
import utils.plotFuncs as pf
from utils.manipulation.manipulator import Manipulator
from time import sleep
import quaternionic as qt
import pybullet_utils.transformations as trans



totaltime = 10

obsPos = np.array([
            [0,0.03,0.66], 
            [-0.25,0.15,0.66], 
        ])

o1 = Obstacle(numObs = 2 , n_dim = 3 , initPos = obsPos , lambda_ = 0.1)

dmp = PositionDMP(N_bfs=100,alpha= 8,cs_alpha=0.5,totaltime = totaltime,cs_tau = 1,obstacle = o1) ## ^ init the DMP class.
quatDMP = QuaternionDMP(alpha = 40, cs_alpha = 1,N_bfs=100,totaltime = totaltime)  ## ^ init the DMP quaternion class

## generate position and orientation trajectory
initPos,initVel,finalPos = np.array([
    [-0.4,0.2,0.69],
    [0,0,0],
    [0.3,-0.2,0.69],
])

initRPY = np.array([0,np.pi,0])
finalRPY = np.array([0.01,np.pi-0.01,0.01])
quatDMP.initQuat = qt.array.from_euler_angles(initRPY).normalized
quatDMP.goalQuat = qt.array.from_euler_angles(finalRPY).normalized

## load environment
kuka = Manipulator(basePosition=[0,0.5,0.65],
                   initEndeffectorPos= initPos,
                   initEndeffectorOrientation=tf.convertTo_pybulletQuat(quatDMP.initQuat))


## loading spheres as obstacles
shift = [0, 0, 0]
meshScale = [0.015, 0.015, 0.015]
visualShapeId = pb.createVisualShape(shapeType=pb.GEOM_MESH,
                                    fileName="sphere_smooth.obj",
                                    meshScale=meshScale)

collisionShapeId = pb.createCollisionShape(shapeType=pb.GEOM_MESH,
                                          fileName="sphere_smooth.obj",
                                          meshScale=meshScale)

pb.loadURDF("table/table.urdf", basePosition=[0,0,0],useFixedBase= True,baseOrientation=pb.getQuaternionFromEuler((0,0,0)) )


pb.createMultiBody(baseMass=1,
                baseInertialFramePosition=[0, 0, 0],
                baseCollisionShapeIndex=collisionShapeId,
                baseVisualShapeIndex=visualShapeId,
                basePosition=obsPos[0])
pb.createMultiBody(baseMass=1,
                baseInertialFramePosition=[0, 0, 0],
                baseCollisionShapeIndex=collisionShapeId,
                baseVisualShapeIndex=visualShapeId,
                basePosition=obsPos[1])



## generate the demo trajectories.
position = tf.generate3DTraj(initPos,initVel,finalPos,dmp.totaltime,dmp.t)
rotation = tf.generateQuaternionTraj(quatDMP.initQuat,quatDMP.goalQuat,quatDMP.t/quatDMP.totaltime)

## train the DMP.
dmp.train(position) ## ^ train the DMP
quatDMP.train(rotation)

print("DMP train complete")
dmp_position = dmp.rollout(position) ## ^ simulate
dmp_quaternion = quatDMP.rollout(rotation)
dmp_pbQuat = tf.convertTo_pybulletQuat(dmp_quaternion)

print("DMP rollout complete")
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
print("pybullet sim starting")    

for i in range(len(dmp.t)):
    
    kuka.setParams()
    velocityControl(i)
    pb_orient.append(kuka.kinematics.linkOrientation)
    pb_position.append(kuka.kinematics.linkPosition)
    
    pb.stepSimulation()

pb_orient = np.array(pb_orient)
pb_position = np.array(pb_position)
pf.plotPosition(dmp.t,position,dmp_position,pb_position)
# pf.plotQuaternions(quatDMP.t,rotation.ndarray,dmp_quaternion.ndarray,pb_orient) ## pass the md array version of the quaternion class matrix
