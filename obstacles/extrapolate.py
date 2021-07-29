## extrapolates the provided trajectory and check if we have to use dynamic obstacle avoidance

import numpy as np

def extrapolateTrajectory(robotPosition,robotVelocity,obstaclePosition,obstacleVelocity,extrapolateTime,dt,tolerance):
    
    time = np.linspace(0,extrapolateTime,int(extrapolateTime/dt) + 1)
    
    # create arrays to store the robot position and obstacle position
    robotExtrapolate = np.empty(shape=(len(time),len(robotPosition)))
    obstacleExtrapolate = np.empty(shape=(len(time),len(obstaclePosition)))
    
    robotExtrapolate[0] = robotPosition
    obstacleExtrapolate[0] = obstaclePosition
    
    for i in range(1,len(time)):
        
        robotExtrapolate[i] = robotExtrapolate[i-1] + robotVelocity*dt
        obstacleExtrapolate[i] = obstacleExtrapolate[i-1] + obstacleVelocity*dt
    
    deltaTrajectory = np.absolute(obstacleExtrapolate - robotExtrapolate) ## difference between the two trajectories at each time step
    
    areTrajClose = deltaTrajectory < tolerance## check if the two trajectories are close enough
    areTrajClose = np.any(areTrajClose)## check if any of the trajectories are close enough
    
    return areTrajClose

if __name__ == '__main__':
    
    robot = np.array([
        [0,0],
        [1,1]
    ])
    obstacle = np.array([
        [1,0],
        [-1,1]
    ])
    
    robotPosition,robotVelocity = robot
    obstaclePosition,obstacleVelocity = obstacle
    
    extrapolateTrajectory(robotPosition,robotVelocity,obstaclePosition,obstacleVelocity,1,0.1,0.5)