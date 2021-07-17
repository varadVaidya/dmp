import numpy as np
import quaternionic as qt

def generate3DTraj(initPos,initVel,finalPos,T,t):
    ## generates minimum jerk trajectory
    ## T = totaltime pf the trajectory
    ## timearray to evaluate the trajectory
    finalVel = np.zeros(3)
    
    boundaryCondition = np.vstack((
        initPos,initVel,finalPos,finalVel
    ))
    
    coeffMatrix = np.array([
        [0,0,0,1],
        [0,0,1,0],
        [T**3,T**2,T,1],
        [3*T**2,2*T,1,0]
    ])
    
    xTrajCoeff,yTrajCoeff,zTrajCoeff = np.linalg.solve(coeffMatrix,boundaryCondition).T
    
    xTraj = np.polyval(xTrajCoeff,t)
    yTraj = np.polyval(yTrajCoeff,t)
    zTraj = np.polyval(zTrajCoeff,t)
    #print(xTrajCoeff)
    posTraj = np.vstack((xTraj,yTraj,zTraj)).T
    return posTraj

def generate2DTraj(initPos,initVel,finalPos,T,t):
    ## generates minimum jerk trajectory
    ## T = totaltime pf the trajectory
    ## timearray to evaluate the trajectory
    finalVel = np.zeros(2)
    
    boundaryCondition = np.vstack((
        initPos,initVel,finalPos,finalVel
    ))
    
    coeffMatrix = np.array([
        [0,0,0,1],
        [0,0,1,0],
        [T**3,T**2,T,1],
        [3*T**2,2*T,1,0]
    ])
    
    xTrajCoeff,yTrajCoeff = np.linalg.solve(coeffMatrix,boundaryCondition).T
    
    xTraj = np.polyval(xTrajCoeff,t)
    yTraj = np.polyval(yTrajCoeff,t)
    #print(xTrajCoeff)
    posTraj = np.vstack((xTraj,yTraj)).T
    return posTraj


def generateQuaternionTraj(initQuat,finalQuat,T):
    
    rotation = qt.interpolation.slerp(initQuat,finalQuat,T)
    return rotation
    
def convertTo_pybulletQuat(quaternion):
    
    ## quaternion is of the class quaternionic
    
    pbQuaternion = np.zeros_like(quaternion.ndarray)
    pbQuaternion[:,0:3] = quaternion.vector
    pbQuaternion[:,3] = quaternion.real
    
    return pbQuaternion

def convertFrom_pybulletQuat(pbQuaternion):
    
    ## pbQuaternion is a list coming from pybullet
    
    quaternion = np.zeros_like(pbQuaternion)
    
    quaternion[0] = pbQuaternion[-1]
    quaternion[1:4] = pbQuaternion[0:3]
    
    return quaternion
    
    
    

if __name__ == '__main__':
    
    what_to_test = 2
    
    if what_to_test == 0:
        
        initPos,initVel,finalPos = np.array([
            [0,0,0],
            [1,0,0],
            [2,3,0],
        ])
        t = np.linspace(0,5,101)
        x,y,z = generate3DTraj(initPos,initVel,finalPos,T = 5,t = t).T
        
        import matplotlib.pyplot as plt
        plt.plot(t,x)
        plt.show()
    
    if what_to_test == 1:
        initPos,initVel,finalPos = np.array([
            [0,0],
            [0,0],
            [2,3]
        ])
        t = np.linspace(0,5,101)
        x,y = generate2DTraj(initPos,initVel,finalPos,T = 5,t = t).T
        
        import matplotlib.pyplot as plt
        plt.plot(t,x)
        plt.show()
    
    if what_to_test == 2:
        q1 = qt.array([1,0,0,0])
        q2 = qt.array([1,0.5,0.5,0.75])
        
        rotation = qt.interpolation.slerp(q1, q2,np.linspace(0,10,101))
        print("rotation",rotation,"\n")
        pbRotation = convertTo_pybulletQuat(rotation)
        print("pbRotation",pbRotation,"\n")