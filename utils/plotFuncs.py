import numpy as np
import matplotlib.pyplot as plt

def plotPosition(time,position,dmp_position,pb_position):
    """
    Plot the position , the DMP trajectory, and the pybullet trajectory, in 3 subplots of matplotlib
    N is the length of the time vector.
    trajPosition is a numpy array of size (N,3). where the 3 coloumns are the X,Y,Z positions.
    dmp_position is a numpy array of size (N,3).
    pbPosition is a numpy array of size (N,3).
    
    """
    
    
    fig,ax = plt.subplots(4,1,sharex= True)

    ax[0].plot(time,position[:,0],label='Demo')
    ax[0].plot(time,dmp_position[:,0],label='DMP')
    ax[0].plot(time,pb_position[:,0],label='pybullet')
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('X')

    ax[1].plot(time,position[:,1],label='Demo')
    ax[1].plot(time,dmp_position[:,1],label='DMP')
    ax[1].plot(time,pb_position[:,1],label='pybullet')
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('Y')

    ax[2].plot(time,position[:,2],label='Demo')
    ax[2].plot(time,dmp_position[:,2],label='DMP')
    ax[2].plot(time,pb_position[:,2],label='pybullet')
    ax[2].set_xlabel('t')
    ax[2].set_ylabel('Z')
    ax[2].legend()

    euclidiean_norm = np.linalg.norm((position - dmp_position) , axis= 1) ## ^ euclidiean norm for position.
    ax[3].plot(time,euclidiean_norm,label='Error Norm')
    ax[3].legend()
    
def plotQuaternions(time,rotation,dmp_quaternion,pb_orient):
    rotationError = np.linalg.norm(rotation - dmp_quaternion,axis=1)
    
    
    figQuat,axQuat = plt.subplots(5,1,sharex=True)

    axQuat[0].plot(time,rotation[:,0],label='Demo')
    axQuat[0].plot(time,dmp_quaternion[:,0],label = "DMP")
    axQuat[0].plot(time,pb_orient[:,3],label = "pybullet")
    axQuat[0].set_xlabel('T(s)')
    axQuat[0].set_ylabel('W')

    axQuat[1].plot(time,rotation[:,1],label='Demo')
    axQuat[1].plot(time,dmp_quaternion[:,1],label='Demo')
    axQuat[1].plot(time,pb_orient[:,0],label = "pybullet")
    axQuat[1].set_ylabel('X')

    axQuat[2].plot(time,rotation[:,2],label='Demo')
    axQuat[2].plot(time,dmp_quaternion[:,2],label='DMP')
    axQuat[2].plot(time,pb_orient[:,1],label = "pybullet")
    axQuat[2].set_ylabel('Y')

    axQuat[3].plot(time,rotation[:,3],label='Demo')
    axQuat[3].plot(time,dmp_quaternion[:,3],label = "DMP")
    axQuat[3].plot(time,pb_orient[:,2],label = "pybullet")
    axQuat[3].set_ylabel('Z')
    axQuat[3].legend()

    axQuat[4].plot(time,rotationError,label='error')
    axQuat[4].legend()  

    plt.show()
        
        