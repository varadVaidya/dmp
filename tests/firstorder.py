import sys,os

path2add = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'dynamical_systems')))
if (not (path2add in sys.path)) :
    sys.path.append(path2add)

print(sys.path)
import numpy as np
import matplotlib.pyplot as plt

from canonical import FirstOrderDynamicalSystems
#from ..dynamical_systems.canonical import FirstOrderDynamicalSystems


## testing the first order canonical system

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

