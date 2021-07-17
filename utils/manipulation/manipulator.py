import pybullet as pb
import pybullet_data
import numpy as np
import sys
sys.path.append( sys.path[0] +'/../..')
from utils.manipulation.kinematics import Kinematics

class Manipulator():
    
    def __init__(self,initJointAngles = None,initEndeffectorPos = None,initEndeffectorOrientation = None):
        ## init kuka robot and plane.
        pb.connect(pb.GUI)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.armID = pb.loadURDF("kuka_iiwa/model.urdf",[0,0,0],useFixedBase=True)
        self.plane = pb.loadURDF("plane.urdf")
        print(self.armID , "arm id")
        ## add gravity
        pb.setGravity(0,0,-9.81)
        pb.setRealTimeSimulation(False)
        
        #define joints  
        self.controlJoints = []
        self.totalJoints = pb.getNumJoints(self.armID)
        self.baseName = pb.getBodyInfo(self.armID)
        
        # self.initJointAngles = initJointAngles
        # self.initEndeffectorPos = initEndeffectorPos
        # self.initEndEffectorOrientation = initEndeffectorOrientation
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
        self.ndof = len(self.controlJoints)
        self.controlZero = [0] * self.ndof
        
        #lower limits for null space
        self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        #upper limits for null space
        self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        #joint ranges for null space
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        #restposes for null space
        self.rp = [0, 0, 0, 0.5 * np.pi, 0, -np.pi * 0.5 * 0.66, 0]
        ## 
        self.kinematics = Kinematics()
        
        self.setInitPos(initJointAngles,initEndeffectorPos,initEndeffectorOrientation)
        self.setParams()
    
    def setInitPos(self,jointAngles = None, endEffectorPos = None,endEffectorOrientation=None):
        
        if endEffectorPos is None and jointAngles is not None:
            
            pb.setJointMotorControlArray(self.armID,self.controlJoints,pb.POSITION_CONTROL,targetPositions =jointAngles)

            for i in range(200):
                pb.stepSimulation()
            
        if jointAngles is None and endEffectorPos is not None:
            
            self.kinematics.inverseKinematics(self.armID,self.endEffectorIndex,endEffectorPos,endEffectorOrientation,
                                                          self.ll,self.ul,self.jr,self.rp)
            
            pb.setJointMotorControlArray(self.armID,self.controlJoints,pb.POSITION_CONTROL,targetPositions =self.kinematics.inv_jointPosition)
            for i in range(200):
                pb.stepSimulation()      
           
    def setParams(self):
        
        ## kinematics that are set here are:
        ##      jointAngles
        ##      end effector position
        ##      geometric jacobian
        self.kinematics.setParams(self.armID,self.endEffectorIndex,self.controlJoints,self.controlZero)
        
        
    # def getForwardKinematics(self):
    #     self.kinematics.endEffectorPos(self.armID,self.endEffectorIndex)

    def getInverseKinematics(self,linkPosition,linkOrientation = None):
        self.kinematics.inverseKinematics(self.armID,self.endEffectorIndex,linkPosition,linkOrientation,self.ll,self.ul,self.jr,self.rp)
    

if __name__ == "__main__":
    
    print("end effector position: FFFFFFF")
    kuka = Manipulator()
    print(kuka.kinematics.geometricJacobian)
        
        
        