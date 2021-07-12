import pybullet as pb
import pybullet_data
import numpy as np
import sys
sys.path.append( sys.path[0] +'/../..')
from utils.manipulation.kinematics import Kinematics

class Manipulator():
    
    def __init__(self):
        ## init kuka robot and plane.
        pb.connect(pb.GUI)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.armID = pb.loadURDF("kuka_iiwa/model.urdf",[0,0,0],useFixedBase=True)
        self.plane = pb.loadURDF("plane.urdf")
        ## add gravity
        pb.setGravity(0,0,-9.81)
        pb.setRealTimeSimulation(False)
        
        #define joints  
        self.controlJoints = []
        self.totalJoints = pb.getNumJoints(self.armID)
        self.baseName = pb.getBodyInfo(self.armID)
        
        # set the end effector link .
        self.endEffectorIndex = -1
        
        ## define the joints that we can control
        for i in range(self.totalJoints):
            
            jointInfo = pb.getJointInfo(self.armID,i)
            
            if jointInfo[2]==0:
                #append the joints we can control
                self.controlJoints.append(i)
                
            if jointInfo[1] == b'lbr_iiwa_joint_7':
                self.endEffectorIndex = i

        self.controlJoints = np.array(self.controlJoints)
        
        ## 
        self.kinematics = Kinematics()
    
    def getForwardKinematics(self):
        self.kinematics.endEffectorPos(self.armID,self.endEffectorIndex)

    def getInverseKinematics(self,linkPosition,linkOrientation = None):
        self.kinematics.inverseKinematics(self.armID,self.endEffectorIndex,linkPosition,linkOrientation)
    

if __name__ == "__main__":
    
    kuka = Manipulator()
    kuka.getForwardKinematics()
    print(kuka.kinematics.linkPosition)      
        
        
        