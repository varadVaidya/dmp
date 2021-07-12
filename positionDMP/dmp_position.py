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
    
    def __init__(self,alpha,cs_alpha,N_bfs = 10,totaltime = 10,cs_tau = 1,n_dim = 3,obstacle = None):
        
        self.N_bfs = N_bfs
        self.alpha = alpha
        self.beta = self.alpha/4
        self.totaltime = totaltime
        self.cs_alpha = cs_alpha
        self.t = np.linspace(0,totaltime,int(totaltime/0.01) + 1)
        self.dt = self.t[1] - self.t[0]
        self.n_dim = n_dim
        # scaling factor.
        self.Dp = np.identity(self.n_dim)
        
        self.cs = CanonicalSystem(alpha = self.cs_alpha,t=self.t,tau= cs_tau)
        
        # centers of the basis functions
        des_activation = np.linspace(0,self.totaltime,self.N_bfs)
        #self.c = np.exp(-self.cs.alpha * np.linspace(0, 1, self.N_bfs))
        self.c = np.exp(-self.cs.alpha*des_activation)
        # variance of the basis functions
        self.h = 1.0/np.gradient(self.c)**2
        
        self.w = np.zeros((self.n_dim,N_bfs))
        
        self.initPos = np.zeros(self.n_dim)
        self.goalPos = np.zeros(self.n_dim)
        
        self.obstacle = obstacle
        self.reset()
    
    def reset(self):
        self.p = self.initPos.copy()
        self.dp = np.zeros(self.n_dim)
        self.ddp = np.zeros(self.n_dim)
        self.cs.reset()
        
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
        
        # weight_sum,psi_sum = np.zeros_like(x),np.zeros_like(x)
        # for i in range(self.N_bfs):
            
        #     psi_i = np.exp(-self.h[i] * (x -self.c[i]) **2)
        #     w_psi = w[i] * psi_i
            
        #     weight_sum += w_psi
        #     psi_sum += psi_i
        
        # approximatedFunc = weight_sum * x / psi_sum
        # return approximatedFunc
        
        def forcing_(xi):
            
            psi = np.exp(-self.h* (xi - self.c) **2)
            
            f_ = (xi * self.w.dot(psi)/psi.sum())

            return  f_
        
        approximatedFunc = np.empty_like(x)
        
        for i in range(len(x)):
            approximatedFunc[i] = forcing_(x[i])
        
        return approximatedFunc
    
    def train(self,position):
        def feature_(xj):
            
            psi = np.exp(-self.h * (xj -self.c) **2)
            
            return xj *psi/psi.sum()
        
        if len(position) != len(self.t):
            raise ValueError("dimensions of the position and time are inconsistent")
        
        self.initPos = position[0]
        self.goalPos = position[-1]
        
        print("goalPos",self.goalPos)
        print("initPos",self.initPos)
        self.x = self.cs.rollout()
                           
        # scaling factor         
        self.Dp = np.diag(self.goalPos - self.initPos)
        Dp_inv = np.linalg.inv(self.Dp)
        
        des_dp = np.gradient(position,axis=0)/self.dt
        des_ddp = np.gradient(des_dp,axis=0)/self.dt
        
        def forcing(i):
            f_i = self.cs.tau**2 * des_ddp[i] - self.alpha * ( self.beta * (self.goalPos - position[i])  - self.cs.tau*des_dp[i] )
            scaled_f_i =  Dp_inv.dot(f_i) 
            return scaled_f_i
        
        A = np.stack(feature_(xj) for xj in self.x)
        #forcingfunc = self.generateRandomForcingFunc()
        forcingfunc = np.stack(forcing(i) for i in range(len(self.t)))
        
        self.w = np.linalg.lstsq(A,forcingfunc,rcond=None)[0].T
        
        # approximatedFunc = self.approximatedFunc(self.x,self.w)
        # self.plotStuff(forcingfunc,approximatedFunc)
        # self.plotGaussians(self.x)
        #plt.plot(self.t,(forcingfunc-approximatedFunc))
        self.trained_p = position
        self.trained_des_dp = des_dp
        self.trained_des_ddp = des_ddp
    
    def step(self,x):
        def forcing_(xi):
            
            psi = np.exp(-self.h* (xi - self.c) **2)
            
            f_ = self.Dp.dot(xi * self.w.dot(psi)/psi.sum() )
            return  f_

        if self.obstacle is None:
            self.ddp = (self.alpha * (self.beta * (self.goalPos - self.p) - self.cs.tau* self.dp ) + forcing_(x) ) * 1/self.cs.tau
        
        if self.obstacle is not None and self.n_dim == self.obstacle.n_dim:
            self.ddp = (self.alpha * (self.beta * (self.goalPos - self.p) - self.cs.tau* self.dp ) 
                        + forcing_(x) + self.obstacle.obstacle_force(self.dp,(self.p - self.obstacle.initPos)) ) * 1/self.cs.tau
            
        self.dp += self.ddp * self.dt
        
        self.p += self.dp * self.dt
        return self.p, self.dp, self.ddp ## ~ returned to plot stuff later on
        
    def rollout(self,position):
        
        self.reset()
        
        #x = self.cs.rollout()
        
        dmp_position = np.empty_like(position)
        
        # ^ note to self. for future reference dt can be passed element by element if dt has to be a numpy vector 
        for i in range(len(self.t)):
            dmp_position[i],_,__ = self.step(self.x[i])
        return dmp_position
        
        
            
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
    
    ## doesnt work anymore. refer examples/cartesianDMP.py
    import matplotlib.pyplot as plt
    
    dmp = PositionDMP(N_bfs=200,alpha= 10,cs_alpha=2,totaltime = 5)
    #position = np.array([np.polyval([1,-2,3,4],dmp.t),np.polyval([-1,2,-3,4],dmp.t),np.polyval([-1,2,-3,4],dmp.t)]).T
    position = np.array([np.sin(dmp.t),np.sin(dmp.t),np.sin(dmp.t)]).T

    # np.savetxt("position.csv",position,fmt="%f",delimiter=",")
    dmp.train(position)
    # np.savetxt("dmp accel.csv",dmp.trained_des_ddp,fmt="%f",delimiter=",")
    # np.savetxt("dmp velocity.csv",dmp.trained_des_dp,fmt="%f",delimiter=",")
    dmp_position = dmp.rollout()
    # np.savetxt("dmp_position.csv",dmp_position,fmt="%f",delimiter=",")
    ## plotting the euclidiean difference between dmp position and the given position
    
    euclidiean_diff = position - dmp_position
    plt.figure()
    plt.plot(dmp.t,euclidiean_diff[:,0],label="error between the trajectories")
    plt.plot(dmp.t,position[:,0],label="given position")
    plt.plot(dmp.t,dmp_position[:,0],label="dmp position")
    plt.legend()
    plt.show()
    
    
    
    
    