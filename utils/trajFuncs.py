import numpy as np

def generate3DTraj(initPos,initVel,finalPos,T,t):
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
    
    return xTraj,yTraj,zTraj

if __name__ == '__main__':
    
    what_to_test = 0
    
    if what_to_test == 0:
        
        initPos,initVel,finalPos = np.array([
            [0,0,0],
            [1,0,0],
            [2,3,0],
        ])
        t = np.linspace(0,5,101)
        x,y,z = generate3DTraj(initPos,initVel,finalPos,T = 5,t = t)
        
        import matplotlib.pyplot as plt
        plt.plot(t,x)
        plt.show()