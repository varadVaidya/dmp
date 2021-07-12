import pybullet as pb
import numpy as np

class Kinematics():
    def __init__(self):
        pass
    
    def endEffectorPos(self,armID,endEffectorIndex):
        endEffector = pb.getLinkState(armID,endEffectorIndex,computeForwardKinematics=True)
        self.linkPosition = endEffector[4]
        self.linkOrientation = endEffector[5]
    
    def inverseKinematics(self,armID,endEffectorIndex,linkPosition,linkOrientation):
        if linkOrientation is not None:
            jointPosition = pb.calculateInverseKinematics(armID,endEffectorIndex,linkPosition,linkOrientation)
            self.inv_jointPosition = jointPosition
        else:
            jointPosition = pb.calculateInverseKinematics(armID,endEffectorIndex,linkPosition)
            self.inv_jointPosition = jointPosition
            