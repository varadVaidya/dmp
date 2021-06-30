import sys
sys.path.append( sys.path[0] +'/..')
import numpy as np
from positionDMP.csSystem import CanonicalSystem 

## placeholder class for testing. used from previous version.
class Gaussian():
    
    def __init__(self,width,center,weight = 1.0):
        
        self.width = width
        self.center = center
        self.weight = weight
    
    def evaluate(self,x):        
        return np.exp(-self.width * (x - self.center) * (x -self.center) )
    def weighted_evaluate(self,x):
        return self.weight * self.evaluate(x)
    
class PositionDMP():
    
    def __init__(self,alpha=1,N_bfs = 10,cs_alpha = None,totaltime = 10,cs_tau = 1):
        
        self.N_bfs = N_bfs
        self.alpha = alpha
        self.beta = self.alpha/4
        self.totaltime = totaltime
        if cs_alpha is None:
            self.cs_alpha = self.alpha/2
        else:
            self.cs_alpha = cs_alpha
        self.t = np.linspace(0,totaltime,int(totaltime/0.01))
        self.dt = np.gradient(self.t)
        
        
        self.cs = CanonicalSystem(alpha = self.cs_alpha,t=self.t,tau= cs_tau)
        
        # centers of the basis functions
        des_activation = np.linspace(0,self.totaltime,self.N_bfs)
        #self.c = np.exp(-self.cs.alpha * np.linspace(0, 1, self.N_bfs))
        self.c = np.exp(-self.cs.alpha*des_activation)
        # variance of the basis functions
        self.h = 1.0/np.gradient(self.c)**2
        
        self.w = np.zeros((N_bfs))
        
        self.initPos = 0
        self.goalPos = 0
    
    ## placeholder function to generate Random Forcing Func
    def generateRandomForcingFunc(self):
        '''
        ? for testing of the method functionApprox::get_weights as the results are not accurate.
        '''
        numBasis = 20
        PSI,values,weight_values = numBasis*[None], numBasis*[None], numBasis*[None]
        
        centers,widths,weights =  np.linspace(0,self.totaltime,numBasis),np.random.randint(1,3,size=(numBasis)) / 4 ,np.random.rand(numBasis)
        weights = np.array(weights)
        weights[0:2],weights[-3:-1] = 0,0
        for i in range(numBasis):
            PSI[i] = Gaussian(widths[i], centers[i],weights[i])
            values[i] = PSI[i].evaluate(self.t)
            weight_values[i] = PSI[i].weighted_evaluate(self.t)
        
        values = np.array(values).T
        weight_values = np.array(weight_values).T
        #print("shape weight_values:", np.shape(weight_values))
        weight_gaussian = np.sum(weight_values,axis=1)
        #print("forcing func", weight_gaussian)
        return weight_gaussian
    
    
    def approximatedFunc(self,x,w):
        
        weight_sum,psi_sum = np.zeros_like(x),np.zeros_like(x)
        for i in range(self.N_bfs):
            
            psi_i = np.exp(-self.h[i] * (x -self.c[i]) **2)
            w_psi = w[i] * psi_i
            
            weight_sum += w_psi
            psi_sum += psi_i
        
        approximatedFunc = weight_sum * x / psi_sum
        return approximatedFunc
    
    def train(self,position):
        def feature_(xj):
            
            psi = np.exp(-self.h * (xj -self.c) **2)
            
            return xj *psi/psi.sum()
        
        if len(position) != len(self.t):
            raise ValueError("dimensions of the position and time are inconsistent")
        
        self.initPos = position[0]
        self.goalPos = position[-1]
        
        x = self.cs.rollout()
                           
        
        des_dp = np.gradient(position)/self.dt
        des_ddp = np.gradient(des_dp)/self.dt
        
        def forcing(i):
            f_i = self.cs.tau**2 * des_ddp[i] - self.alpha * ( (self.beta * (self.goalPos - position[i]) ) - self.cs.tau*des_dp[i] )
            return f_i
        
        A = np.stack(feature_(xj) for xj in x)
        #forcingfunc = self.generateRandomForcingFunc()
        forcingfunc = np.stack(forcing(i) for i in range(len(self.t)))
        self.w = np.linalg.lstsq(A,forcingfunc,rcond=None)[0]
        #print(self.w)
        approximatedFunc = self.approximatedFunc(x,self.w)
        self.plotStuff(forcingfunc,approximatedFunc)
        self.plotGaussians(x)
    def plotGaussians(self,x):
        
        gaussianMatrix = np.empty(( self.N_bfs, len(self.t) ))
        plt.figure()
        plt.title('Gaussians')
        
        for i in range(self.N_bfs):
            gaussianMatrix[i] = np.exp(-self.h[i] * (x -self.c[i]) **2)
            plt.plot(self.t,gaussianMatrix[i])
        plt.show()
        
    def plotStuff(self,forcingfunc,approximatedFunc):
        plt.plot(self.t,forcingfunc,label = "forcing func")
        plt.plot(self.t,approximatedFunc,label="approximated func")
        plt.legend()
        plt.show()
        
if __name__ =="__main__":
    
    import matplotlib.pyplot as plt
    dmp = PositionDMP(N_bfs=30)
    position = np.polyval([1,6,11,6],dmp.t)
    dmp.train(position)
    
    
    