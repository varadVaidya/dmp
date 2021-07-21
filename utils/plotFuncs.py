import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plotPosition(time,position,dmp_position,pb_position = None):
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
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('X')

    ax[1].plot(time,position[:,1],label='Demo')
    ax[1].plot(time,dmp_position[:,1],label='DMP')
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('Y')

    ax[2].plot(time,position[:,2],label='Demo')
    ax[2].plot(time,dmp_position[:,2],label='DMP')
    ax[2].set_xlabel('t')
    ax[2].set_ylabel('Z')

    if pb_position is not None:
        ax[0].plot(time,pb_position[:,0],label='pybullet')  
        ax[1].plot(time,pb_position[:,1],label='pybullet')
        ax[2].plot(time,pb_position[:,2],label='pybullet')
        
    euclidiean_norm = np.linalg.norm((position - dmp_position) , axis= 1) ## ^ euclidiean norm for position.
    ax[3].plot(time,euclidiean_norm,label='Error Norm')
    
    ax[2].legend()
    ax[3].legend()
    
    plt.show()
    
def plotQuaternions(time,rotation,dmp_quaternion,pb_orient = None):
    rotationError = np.linalg.norm(rotation - dmp_quaternion,axis=1)
    
    
    figQuat,axQuat = plt.subplots(5,1,sharex=True)

    axQuat[0].plot(time,rotation[:,0],label='Demo')
    axQuat[0].plot(time,dmp_quaternion[:,0],label = "DMP")
    axQuat[0].set_xlabel('T(s)')
    axQuat[0].set_ylabel('W')

    axQuat[1].plot(time,rotation[:,1],label='Demo')
    axQuat[1].plot(time,dmp_quaternion[:,1],label='Demo')
    axQuat[1].set_ylabel('X')

    axQuat[2].plot(time,rotation[:,2],label='Demo')
    axQuat[2].plot(time,dmp_quaternion[:,2],label='DMP')
    axQuat[2].set_ylabel('Y')

    axQuat[3].plot(time,rotation[:,3],label='Demo')
    axQuat[3].plot(time,dmp_quaternion[:,3],label = "DMP")
    axQuat[3].set_ylabel('Z')

    if pb_orient is not None:
        axQuat[0].plot(time,pb_orient[:,3],label = "pybullet")
        axQuat[1].plot(time,pb_orient[:,0],label = "pybullet")
        axQuat[2].plot(time,pb_orient[:,1],label = "pybullet")
        axQuat[3].plot(time,pb_orient[:,2],label = "pybullet")
        
    axQuat[4].plot(time,rotationError,label='error')
    axQuat[3].legend()
    axQuat[4].legend()  

    plt.show()
        
def animatePositionDMP2D(time,position,dmp_position,obstaclePosition = None,saveVideo = False):
    
    if obstaclePosition is None:
        
        time_min,time_max = np.min(time),np.max(time)
        
        xmin,xmax = np.min(position[:,0]),np.max(position[:,0])
        ymin,ymax = np.min(position[:,1]),np.max(position[:,1])
        
        xmax += np.absolute(xmax/5)
        ymax += np.absolute(ymax/5)
        
        xmin -= np.absolute(xmin/5)
        ymin -= np.absolute(ymin/5)
        
        fig = plt.figure()
        ax = plt.axes(xlim=(xmin,xmax),ylim=(ymin,ymax))
        
        line, = ax.plot([], [], lw=2) ## line 1 is provided traj.
        line2, = ax.plot([], [] , lw=2) ## line 2 is the dmp traj.
        # line3, = ax.plot([], [] , lw=2)
        
        posX,posY,dmpPosX,dmpPosY = [],[],[],[]
        
        def init():
            line.set_data(position[:,0],position[:,1] )
            line2.set_data(dmp_position[:,0],dmp_position[:,1])
            
            return line,line2
        
        def animate(i):
            
            posX.append(position[i,0])
            posY.append(position[i,1])
            dmpPosX.append(dmp_position[i,0])
            dmpPosY.append(dmp_position[i,1])
            
            line.set_data(posX,posY)
            line2.set_data(dmpPosX,dmpPosY)
            
            return line,line2
        
        ani = animation.FuncAnimation(fig, animate, init_func=init,interval = 1,frames= len(time),repeat=False)
        plt.show()    
    
    if obstaclePosition is not None:
        
        time_min,time_max = np.min(time),np.max(time)
        
        xmin,xmax = np.min(position[:,0]),np.max(position[:,0])
        ymin,ymax = np.min(position[:,1]),np.max(position[:,1])
        
        xmax += np.absolute(xmax/5)
        ymax += np.absolute(ymax/5)
        
        xmin -= np.absolute(xmin/5)
        ymin -= np.absolute(ymin/5)
        
        fig = plt.figure()
        ax = plt.axes(xlim=(xmin,xmax),ylim=(ymin,ymax))
        
        line, = ax.plot([], [], lw=2) ## line 1 is provided traj.
        line2, = ax.plot([], [] , lw=2) ## line 2 is the dmp traj.
        line3, = ax.plot([], [] , lw=2) ## line 3 is the obstacle traj.])
        # line3, = ax.plot([], [] , lw=2)
        
        posX,posY,dmpPosX,dmpPosY = [],[],[],[]
        obsX,obsY = [],[]
        
        def init():
            line.set_data(position[:,0],position[:,1] )
            line2.set_data(dmp_position[:,0],dmp_position[:,1])
            line3.set_data(obstaclePosition[:,0],obstaclePosition[:,1])
            
            return line,line2,line3
        
        def animate(i):
            
            posX.append(position[i,0])
            posY.append(position[i,1])
            dmpPosX.append(dmp_position[i,0])
            dmpPosY.append(dmp_position[i,1])
            
            obsX.append(obstaclePosition[i,0])
            obsY.append(obstaclePosition[i,1])
            
            line.set_data(posX,posY)
            line2.set_data(dmpPosX,dmpPosY)
            line3.set_data(obsX,obsY)
            
            return line,line2,line3
        
        ani = animation.FuncAnimation(fig, animate, init_func=init,interval = 1,frames= len(time),repeat=False)
        
        if saveVideo:    
            writerVideo = animation.writers['ffmpeg'](fps=30)
            ani.save('dmp_position.mp4',writerVideo)
         
        plt.show()

def animatePositionDMP3D(time,position,dmp_position,obstaclePosition = None,saveVideo = False):
    
    import mpl_toolkits.mplot3d.axes3d as p3
    
    if obstaclePosition is None:
        
        time_min,time_max = np.min(time),np.max(time)
        
        xmin,xmax = np.min(position[:,0]),np.max(position[:,0])
        ymin,ymax = np.min(position[:,1]),np.max(position[:,1])
        zmin,zmax = np.min(position[:,2]),np.max(position[:,2])
        
        xmax += np.absolute(xmax/5)
        ymax += np.absolute(ymax/5)
        zmax += np.absolute(zmax/5)
        
        xmin -= np.absolute(xmin/5)
        ymin -= np.absolute(ymin/5)
        zmin -= np.absolute(zmin/5)
        
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        
        line, = ax.plot([], [], [], lw=2)
        line2, = ax.plot([], [], [] , lw=2)
        
        posX,posY,posZ = [],[],[]
        dmpPosX,dmpPosY,dmpPosZ = [],[],[]
        
        ax.set_xlim3d(xmin,xmax)
        ax.set_ylim3d(ymin,ymax)
        ax.set_zlim3d(zmin,zmax)
        
        def init():
            line.set_data([],[] )
            line.set_3d_properties([])
            
            line2.set_data([],[] )
            line2.set_3d_properties([])
            
            return line,line2
        
        def animate(i):
            posX.append(position[i,0])
            posY.append(position[i,1])
            posZ.append(position[i,2])
            
            dmpPosX.append(dmp_position[i,0])
            dmpPosY.append(dmp_position[i,1])
            dmpPosZ.append(dmp_position[i,2])
            
            line.set_data(posX,posY)
            line.set_3d_properties(posZ)
            
            line2.set_data(dmpPosX,dmpPosY)
            line2.set_3d_properties(dmpPosZ)
            
            return line,line2
        
        ani = animation.FuncAnimation(fig, animate, init_func=init,interval = 1,frames= len(time),repeat=False)
        
        plt.show()
        
           
    if obstaclePosition is not None:
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        
        xmin,xmax = np.min(position[:,0]),np.max(position[:,0])
        ymin,ymax = np.min(position[:,1]),np.max(position[:,1])
        zmin,zmax = np.min(position[:,2]),np.max(position[:,2])
        
        xmax += np.absolute(xmax/5)
        ymax += np.absolute(ymax/5)
        zmax += np.absolute(zmax/5)
        
        xmin -= np.absolute(xmin/5)
        ymin -= np.absolute(ymin/5)
        zmin -= np.absolute(zmin/5)
        
        line, = ax.plot([], [], [], lw=2) ## line 1 is provided traj.
        line2, = ax.plot([], [] , [], lw=2) ## line 2 is the dmp traj.
        line3, = ax.plot([], [] , [], lw=2) ## line 3 is the obstacle traj.
        
        
        posX,posY,posZ = [],[],[]
        dmpPosX,dmpPosY,dmpPosZ = [],[],[]
        obsPosX,obsPosY,obsPosZ = [],[],[]
        
        ax.set_xlim3d(xmin,xmax)
        ax.set_ylim3d(ymin,ymax)
        ax.set_zlim3d(zmin,zmax)
        
        def init():
            line.set_data([],[] )
            line.set_3d_properties([])
            
            line2.set_data([],[] )
            line2.set_3d_properties([])
            
            line3.set_data([],[] )
            line3.set_3d_properties([])
            
            return line,line2,line3
        
        def animate(i):
            posX.append(position[i,0])
            posY.append(position[i,1])
            posZ.append(position[i,2])
            
            dmpPosX.append(dmp_position[i,0])
            dmpPosY.append(dmp_position[i,1])
            dmpPosZ.append(dmp_position[i,2])
            
            obsPosX.append(obstaclePosition[i,0])
            obsPosY.append(obstaclePosition[i,1])
            obsPosZ.append(obstaclePosition[i,2])
            
            line.set_data(posX,posY)
            line.set_3d_properties(posZ)
            
            line2.set_data(dmpPosX,dmpPosY)
            line2.set_3d_properties(dmpPosZ)
            
            line3.set_data(obsPosX,obsPosY)
            line3.set_3d_properties(obsPosZ)
            
            return line,line2,line3
        
        ani = animation.FuncAnimation(fig, animate, init_func=init,interval = 1,frames= len(time),repeat=False)
        
        plt.show()
    
        
    
    
    