## canonical system

import numpy as np

class CanonicalSystem():
    
    def __init__(self,alpha,t,tau):
        self.alpha = alpha
        self.dt = np.gradient(t)[0]
        self.tau = tau
        self.reset()
    
    def step(self):
        
        self.x += self.dt * -self.alpha*self.x / self.tau
        
        
    def rollout(self):
        '''
        solve the canonical system
        '''
        rolloutArray = np.empty_like(t)
        for i in range(len(t)):
            rolloutArray[i] = self.x
            self.step()   
        
        return rolloutArray    
    
    def reset(self):
        self.x = 1.0
if __name__ == "__main__":
    
    t = np.linspace(0,5,100)
    cs = CanonicalSystem(alpha= 4,t = t,tau=1)
    rollout = cs.rollout()
    print(type(rollout))
    print(rollout)    
