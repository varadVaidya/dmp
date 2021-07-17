import sys
sys.path.append( sys.path[0] +'/..')
import numpy as np
from positionDMP.csSystem import CanonicalSystem
import quaternionic as qt

class QuaternionDMP():
    
    def __init__(self,alpha,cs_alpha,N_bfs = 10,totaltime = 10,cs_tau = 1):
        
        self.N_bfs = N_bfs
        self.alpha = alpha
        self.beta = self.alpha/4
        self.totaltime = totaltime
        self.cs_alpha = cs_alpha
        self.t = np.linspace(0,totaltime,int(totaltime/0.01) + 1)
        self.dt = self.t[1] - self.t[0]
        
        self.cs = CanonicalSystem(alpha = self.cs_alpha,t=self.t,tau= cs_tau)
        #self.x = self.cs.rollout()
        
        # centers of the basis functions
        des_activation = np.linspace(0,self.totaltime,self.N_bfs)
        #self.c = np.exp(-self.cs.alpha * np.linspace(0, 1, self.N_bfs))
        self.c = np.exp(-self.cs.alpha*des_activation)
        # variance of the basis functions
        self.h = 1.0/np.gradient(self.c)**2
        
        self.w = np.zeros((3,N_bfs))
        
        self.Do = np.identity(3)
        
        self.initQuat = qt.array([1,0,0,0])
        self.goalQuat = qt.array([1,0,0,0])
        
        self.reset()
    
    def reset(self):
        self.q = self.initQuat.copy()
        self.qdot = qt.array([0,0,0,0])
        self.eta = qt.array([0,0,0,0]) ## ^ eta belongs to R^3 and is treated as a quaternion with scalar part as 0.
        self.etadot = qt.array([0,0,0,0])
        self.cs.reset()
    
    def normaliseQuat(self):
        self.q = self.q.normalized
        # self.eta = self.eta.normalized
        # self.etadot = self.etadot.normalized
        
        
    def step(self,xi):
        def forcing_(xi):
            
            psi = np.exp(-self.h* (xi - self.c) **2)
        
            f_ = self.Do.dot(xi * self.w.dot(psi)/psi.sum() )
            
            return f_
          
        self.etadot = self.alpha * ( self.beta * 2 * np.log(self.goalQuat * self.q.conjugate() )  - self.eta) + qt.array.from_vector_part(forcing_(xi))
        ## intergrate etadot to eta
        self.etadot /= self.cs.tau
        
        self.eta += self.etadot * self.dt 
        self.eta /= self.cs.tau
        
        self.q = np.exp((self.dt/2) * (self.eta/self.cs.tau)) * self.q
        
        self.normaliseQuat()
        
        return self.q

    def train(self,rotation):
        def feature_(xj):
            
            psi = np.exp(-self.h * (xj -self.c) **2)
            
            return xj *psi/psi.sum()
        
        if len(rotation) != len(self.t):
            raise ValueError("dimensions of the rotation and time are inconsistent")
        
        # self.initQuat = rotation[0]
        # self.initQuat = rotation[-1]
        
        rotationError = 2 * np.log(self.goalQuat * self.initQuat.conjugate() )
        #rotationError = self.goalQuat * self.initQuat.conjugate() 
        self.Do = np.diag(rotationError.vector)

        DO_inv = np.linalg.inv(self.Do)
        
        des_dq = []
        des_dq.append(qt.array([0,0,0,0]))
        
        for i in range(len(rotation) -1 ):
            eta = 2 * np.log(rotation[i+1] * rotation[i].conjugate()) / self.dt
            des_dq.append(eta)
        
        des_dq = qt.array(np.array(des_dq))
        
        des_ddq = qt.array(np.gradient(des_dq,axis=0)/self.dt)
        
        self.x = self.cs.rollout()
        
        def forcing(i):
            betaTerm = 2 * np.log(self.goalQuat * rotation[i].conjugate())
            f_i = self.cs.tau**2 * des_ddq[i].vector - self.alpha * ( self.beta * betaTerm.vector  - self.cs.tau*des_dq[i].vector )
            scaled_f_i =  DO_inv.dot(f_i) 
            return scaled_f_i

        A = np.stack(feature_(xj) for xj in self.x)
        
        forcingfunc = np.stack(forcing(i) for i in range(len(self.t)))
        
        
        self.w = np.linalg.lstsq(A,forcingfunc,rcond=None)[0].T

    def rollout(self,rotation):
        
        self.reset()
        
        dmp_rotation = np.empty_like(rotation)
        
        for i in range((len(self.t))):
            dmp_rotation[i] = self.step(self.x[i])
        
        return dmp_rotation
        
## ! testing QuaternionDMP::step()

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    quatDMP = QuaternionDMP(alpha = 10, cs_alpha = 2,N_bfs=100,totaltime = 5)
    quatDMP.initQuat = qt.array([1,0,-1,0]).normalized
    quatDMP.goalQuat = qt.array([1,2,1,-1]).normalized
    
    print("initQuat",quatDMP.initQuat)
    print("goalQuat",quatDMP.goalQuat)
    rotation = qt.interpolation.slerp(quatDMP.initQuat,quatDMP.goalQuat,quatDMP.t/quatDMP.totaltime)
    # rotation = rotation.normalized
    print("rotation[0]",rotation[0])
    print("rotation[-1]",rotation[-1])
    quatDMP.train(rotation)
    # quatDMP.step()
    # quatDMP.step()
    
    # for i in range(len(quatDMP.t)):
    #     quatDMP.step()
    #     plotQ.append(quatDMP.q)
    
    plotQ = quatDMP.rollout(rotation)
    plotQ = np.array(plotQ).T
    print(plotQ.shape)
    
    rotation = rotation.ndarray
    
    
    
    
    fig , ax = plt.subplots(4,1,sharex = True)
    
    ax[0].plot(quatDMP.t,plotQ[0])
    ax[0].plot(quatDMP.t,rotation[:,0])
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('w')
    
    ax[1].plot(quatDMP.t,plotQ[1])
    ax[1].plot(quatDMP.t,rotation[:,1])
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('x')
    
    ax[2].plot(quatDMP.t,plotQ[2])
    ax[2].plot(quatDMP.t,rotation[:,2])
    ax[2].set_xlabel('t')
    ax[2].set_ylabel('y')
    
    ax[3].plot(quatDMP.t,plotQ[3],label = 'DMP')
    ax[3].plot(quatDMP.t,rotation[:,3],label = "demo")
    ax[3].set_xlabel('t')
    ax[3].set_ylabel('z')
    ax[3].legend()
    
    plt.show()
    
    print(plotQ[:,-1])
    print(quatDMP.goalQuat.normalized)