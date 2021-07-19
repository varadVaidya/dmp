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

kuka = Manipulator()
## generate time array to extract position and rotation data
totaltime = 60.0
t = np.linspace(0,totaltime,int(totaltime/0.01) + 1)

# load the data
data = np.load("utils/traj_target.npy")
trimmedData = data[1:len(t)+1].copy()

# convert to the format expected by DMP system
position = trimmedData[:,0:3].copy()
quatRotation = trimmedData[:,6:10].copy()

rotation = qt.array(quatRotation)
rotation = rotation.normalized

## init the DMP class for both position and quaternion
dmp = PositionDMP(N_bfs=1000,alpha= 20,cs_alpha=0.3,totaltime = totaltime,cs_tau = 1) ## ^ init the DMP class.
quatDMP = QuaternionDMP(alpha = 40, cs_alpha = 0.3,N_bfs=1000,totaltime = totaltime)  ## ^ init the DMP quaternion class
quatDMP.initQuat = qt.array(quatRotation[0]).normalized
quatDMP.goalQuat = qt.array(quatRotation[-1]).normalized

print("initQuat",quatDMP.initQuat)
print("finalQuat",quatDMP.goalQuat)


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
    
    velocityControl(i)
    pb_orient.append(kuka.kinematics.linkOrientation)
    pb_position.append(kuka.kinematics.linkPosition)
    pb.stepSimulation()
    sleep(0.01)

pb_orient = np.array(pb_orient)
pb_position = np.array(pb_position)

pf.plotPosition(dmp.t,position,dmp_position,pb_position)
pf.plotQuaternions(quatDMP.t,rotation.ndarray,dmp_quaternion.ndarray,pb_orient) ## pass the md array version of the quaternion class matrix

