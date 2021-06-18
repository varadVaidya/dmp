import numpy as np

class Gaussian():
    
    def __init__(self,width,center,weight = 1.0):
        
        self.width = width
        self.center = center
        self.weight = weight
    
    def evaluate(self,x):        
        return np.exp(-self.width * (x - self.center) * (x -self.center) )
    def weighted_evaluate(self,x):
        return self.weight * self.evaluate(x)

class functionApprox():
    
    ## assuming that we have forcing function as numpy array...
    
    @staticmethod
    def create_Gaussian_basis(numBasis,firstorderSystem):
        
        ## init the basis functions 
        center = numBasis *[None]
        width = numBasis * [None]
        weight = numBasis * [None]
        
        PSI = numBasis * [None]
        
        des_activation = np.linspace(0,firstorderSystem.totaltime,numBasis)
        
        ## calculate the centers
        for i in range(numBasis):
            #center[i] = np.exp(-firstorderSystem.alpha * des_activation[i])
            center[i] = des_activation[i]
            #width[i] = numBasis**1.2 / center[i] / firstorderSystem.alpha
        
        ## calculate width
        for j in range(numBasis-1):
            width[j] = (center[j+1] - center[j]) * (center[j+1] - center[j])
        
        width[-1] = width[-2]               
        # print("centers",center)
        # print("width",width)
        
        ## fill the gaussina class
        
        gaussian_values = numBasis * [None]
        for i in range(numBasis):
            
            PSI[i] = Gaussian(width[i], center[i])
            gaussian_values[i] = PSI[i].evaluate(firstorderSystem.timearray)
        
        
        
if __name__ == '__main__':
    
    from dynamical_systems.canonical import FirstOrderDynamicalSystems  
    phaseSystem = FirstOrderDynamicalSystems(alpha = 2)
    
    functionApprox.create_Gaussian_basis(10,phaseSystem)
    
    
            
        
        