"""
implementation of canonical systems
both the first order and the second order systems are implemented
"""

import numpy as np
import matplotlib.pyplot as plt

class FirstOrderDynamicalSystems():
    
    def __init__(self,alpha,beta = 1.0,timeconstant = 1.0,system = 0,dt = 0.001,x0 = 1,t0 =0, goal = 0):
        """
        init the class and set its properties

        Args: \n
            alpha (float): decay contant of the system. \n
            timeconstant (float): leads to faster or convergence to the attractor. Defaults to 1.0 \n
            system (int): integer to set the flag of what system we will use. Defaults to 0.
                        0 == exponential decay
                        1 == sigmoid
                        2 == constant velocity \n
            beta(float): sigmoid function value. Defaults to 1.0 \n
            dt (float): time step for the canonical system simulation. Defaults to 0.001. \n
            x0 (float): inital state of the system. Defaults to 1. \n
            t0 (float): inital time \n
            goal (int): goal state of the system. Defaults to 0. \n
        """
        ## set the parameters in the class
        self.alpha = alpha
        self.dt = dt
        self.x0 = x0
        self.t0 = t0
        self.goal = goal
        self.system = system
        self.timeconstant = timeconstant
        self.beta = beta
    
    def integrate(self,x,dt):
        """ this function will apply the euler step to the canonical system according to the type of the system

        Args: \n
            x (float): current system of the system \n
            dt (float): [description]. Defaults to self.dt. \n
            
        Returns: \n
            x (float): current system of the system \n
            t (float): current time of the system \n
        """
        ## decide the velocity profile of the system according to the value set in self.system.
        if self.system == 0:
            dx = -self.alpha * (x - self.goal) * 1/self.timeconstant  ## based on the system rule. set to exponential decay.
        
        if self.system == 1:
            dx = self.alpha * x *(self.beta - (x - self.goal))
        xnext = x + dx * dt ## euler step for getting cuuent state
        #tnext = t + dt ## change the current time.
        
        return xnext
    
    def simulation(self,totaltime = 10):
        """plan is to simulate the system for the provided simulaiton time.
        Args:
            totaltime (float): total time of simulation to be executed. Defaults to 10.

        Returns:
            statearray (numpy array): numpy array of the all the states after the simulation.
            timearray (numpy array): numpy array of the time state. 
        """
        ## plan is to simulate the system for the provided simulaiton time.
        
        timearray = np.linspace(self.t0,totaltime,int(totaltime/self.dt)) ## setup the time array for which simulation will be done
        
        statearray = [] ## array to store the state variables
        currentstate = self.x0 ## define current state
        
        
        for i in range(len(timearray)): ## loop through time array
        
            statearray.append(currentstate)
            xnext = self.integrate(currentstate,self.dt) ## euler integration
            currentstate = xnext ## set the next state as current state for the next loop
        
        statearray = np.array(statearray) ## convert to numpy array for plotting
        
        return statearray,timearray
    
    
    
if __name__ == "__main__":
    
    ## testing the first order canonical system
    
    cs1 = FirstOrderDynamicalSystems(alpha=1,x0 = 4,goal = 1.5) # ini the class
    states1,times = cs1.simulation(totaltime = 10) 
    cs2 = FirstOrderDynamicalSystems(alpha=1,x0 = 4,timeconstant = 0.3) # ini the class
    states2,times = cs2.simulation(totaltime = 10)
    cs3 = FirstOrderDynamicalSystems(alpha=1,x0 = 4,beta = 1.5,system = 1)
    states3,times = cs3.simulation(totaltime = 10)
    
    plt.plot(times,states1,label = "alpha = 1,goal = 0,timeconstant = 0")
    plt.plot(times,states2,label = "alpha = 1,goal =0,timeconstant = 0.7")
    plt.plot(times,states3,label = "alpha = 1,goal = 0 , sigmoid")
    plt.legend()
    plt.show()
    
    
    
        
        
    
    
        
        
        
    
