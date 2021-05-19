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
            goal (float): goal state of the system. Defaults to 0. \n
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



class SecondOrderDynamicalSystem():
    
    def __init__(self,alpha,beta = None ,timeconstant = 1.0,dt = 0.001,x0 = np.array([0,1]),t0 =0, goal = 0):
        """init the SecondOrderDynamicalSystem class \n

        Args: \n 
            alpha (float): decay contant of the system. \n
            beta (float): decay constant of the system. Defaults to alpha/4 for critically damped system. \n
            timeconstant (float): leads to faster or convergence to the attractor. Defaults to 1.0 \n. Defaults to 1.0. \n
            dt (float): time step for the canonical system simulation. Defaults to 0.001. \n
            x0 (numpy array size 2*1): inital state of the system. Defaults to [0 1]. \n
            t0 (float): inital time. Defaults to 0. \n
            goal (float): goal state of the system. Defaults to 0. \n
        """
        
        ## set the parameters in the class
        self.alpha = alpha
        if beta is not None: ## if beta is specified keep the value as provided
            self.beta = beta
        else: ## else default back to alpha/4
            self.beta = self.alpha / 4
        self.timeconstant = timeconstant
        self.dt = dt
        self.x0 = x0
        self.t0 = t0
        self.goal = goal
        
    def integrate(self,x,dt):
        """ performs eu

        Args:
            x (numpy array size 2*1): initial state of the system \n
            dt (float): time step for the canonical system simulation. Defaults to 0.001. \n

        Raises:
            Exception: if the state vector is of wrong size. will raise error to fix it

        Returns:
            xnext [numpy array]: next state in the simulation
        """
        ## the current state of the system is assumed to be a 2*1 vector, where the second order system is reduced to 2 first order equations.
        
        if x.shape != (2,):
            raise Exception("Wrong dimensions for the state vector")
        
        z,y = x ## unpacking the state vector
        
        dz = ( self.alpha * (self.beta * (self.goal - y) - z ) )/self.timeconstant  ## define the derivative according to the system rule
        dy = z/self.timeconstant  ## define the derivative according to the system rule
        
        znew = z + dz * self.dt  ## euler step
        ynew = y + dy * self.dt  ## euler step
        
        xnext = np.array([znew,ynew]) ##pack the state vector in numpy array
        
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
    
    what_to_test = 0
    """
    flag to test classes in this py file:
    if what_to_test is 0 test first order.
    if what_to_test is 1 test second order.
    """
    ## testing the first order canonical system
    
    if what_to_test == 0:
        cs1 = FirstOrderDynamicalSystems(alpha=1,x0 = 4,goal = 1.5) # ini the class
        states1,times = cs1.simulation(totaltime = 10) 
        cs2 = FirstOrderDynamicalSystems(alpha=1,x0 = 4,timeconstant = 0.3) # ini the class
        states2,times = cs2.simulation(totaltime = 10)
        cs3 = FirstOrderDynamicalSystems(alpha=1,x0 = 4,beta = 1.5,system = 1)
        states3,times = cs3.simulation(totaltime = 10)
        plt.title("First Order Canonical System")
        plt.plot(times,states1,label = "alpha = 1,goal = 0,timeconstant = 0")
        plt.plot(times,states2,label = "alpha = 1,goal =0,timeconstant = 0.7")
        plt.plot(times,states3,label = "alpha = 1,goal = 0 , sigmoid")
        plt.legend()
        plt.show()
    

    if what_to_test == 1:
        ## testing the second order canonical system
        
        secondCS = SecondOrderDynamicalSystem(alpha = 3,goal=4)
        states,times = secondCS.simulation(totaltime = 10)
        #print(states)
        
        plt.plot(times,states[:,0],label = "velocity")
        plt.plot(times,states[:,1],label = "position")
        plt.legend()
        plt.show()
    
    
    
    
        
        
    
    
        
        
        
    
