import pybullet as pb
import numpy as np

class Kinematics():
    def __init__(self):        
        pass
    
    def jointState(self):
        joint = pb.getJointStates(self.armID,self.controlJoints)
        
        jointAngles = [ i[0] for i in joint]
        jointVelocities = [ i[1] for i in joint]
        jointReactionForces = [ i[2] for i in joint]
        
        self.jointAngles = jointAngles
        self.jointVelocities = jointVelocities
        self.jointReactionForces = jointReactionForces
                
        
    def endEffectorPos(self):
        
        endEffector = pb.getLinkState(self.armID,self.endEffectorIndex,computeForwardKinematics=True)
        self.linkPosition = endEffector[4]
        self.linkOrientation = endEffector[5]
        
    
    def getGeometricJacobian(self):
        endEffector = pb.getLinkState(self.armID,self.endEffectorIndex)
        
        jointAngles = self.jointAngles
        
        linJac,angJac = pb.calculateJacobian(self.armID,self.endEffectorIndex,endEffector[2],jointAngles,self.controlZero,self.controlZero)

        self.geometricJacobian = np.vstack((linJac,angJac))
        self.geometricJacobianInv = np.linalg.pinv(self.geometricJacobian)
        
    
    ## utils functions
    def inverseKinematics(self,armID,endEffectorIndex,linkPosition,linkOrientation,ll,ul,jr,rp):
        if linkOrientation is not None:
            jointPosition = pb.calculateInverseKinematics(armID,endEffectorIndex,linkPosition,linkOrientation)
            self.inv_jointPosition = jointPosition
        else:
            jointPosition = pb.calculateInverseKinematics(armID,endEffectorIndex,linkPosition)
            self.inv_jointPosition = jointPosition

    def setParams(self,armID,endEffectorIndex,controlJoints,controlZero):
        
        self.armID = armID
        self.endEffectorIndex = endEffectorIndex
        self.controlJoints = controlJoints
        self.controlZero = controlZero           
        
        self.jointState()
        self.endEffectorPos()
        self.getGeometricJacobian()
        
    
    
    
        
               