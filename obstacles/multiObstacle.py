## point obstacle container class
## notations used here are from the paper refered in the README
import numpy as np
class Obstacle():
    ## will contain all the obstacle info necessary:
    ## also has the paramerters to define the dynamic potential field
    ## obstacle can also move with time
    def __init__(self,n_dim = 3,numObs = 1,initPos = None ,initVel = None,lambda_ =1):
        
        self.n_dim = n_dim ## dimension in which the obstacle exists
        self.numObs = numObs ## number of obstacles
        
        if initPos is None:
            self.initPos = np.zeros((self.numObs,self.n_dim)) ## the initial position of the obstacle
            self.initVel = np.zeros((self.numObs,self.n_dim)) ## initial velocity of of the obstacle
            
        if initPos is not None:
            
            self.initPos = initPos
            
            if initVel is None:
                self.initVel = np.zeros((self.numObs,self.n_dim)) ## the initial velocity of the obstacle
            
            else:
                self.initVel = initVel
        
        self.currentPos = self.initPos.copy() ## current position of the obstacle
        self.currentVel = self.initVel.copy() ## current velocity of the obstacle
        ## parameters for the dynamic potential field.
        self.speed = np.zeros(self.n_dim) ## speed of the obstacle.
        self.lambda_ = lambda_
        self.beta = 3
        self.obstaclePos = []
        
        
    
    def get_distance(self,X):
        ## distance between obstacle and position of the end effector
        
        if self.numObs == 1:
            distance = np.linalg.norm(X)
        else:
            distance = np.linalg.norm(X,axis=1)
        return distance
        
    def get_cosine_angle(self,V,X):
        ## get the angle between the velocity vector and the position vector of the end effector relative to the obstacle.
        numerator = np.sum(V * X, axis=1)
        mod_V = np.linalg.norm(V) # modulus of the velocity vector
        
        denominator = mod_V * self.get_distance(X)
        
        cosine_angle = numerator/denominator
        
        return cosine_angle
    
    def grad_P(self,X):
        
        delta_P = (X)/self.get_distance(X)[:,None]
        return delta_P
    
    def grad_cosine_angle(self,V,X):
        
        distance = self.get_distance(X)
        mod_V = np.linalg.norm(V)
        vTx = np.sum(V * X, axis=1)
        delta_P = self.grad_P(X)
        
        distanceDotV = distance[:,np.newaxis] * V[np.newaxis,:]
        
        Vtx_Dot_DeltaP = (delta_P.T * vTx).T
        numerator = distanceDotV - Vtx_Dot_DeltaP
        
        denominator = mod_V * distance **2
        
        delta_cos = numerator/denominator[:,None]
        
        return delta_cos

    def calculateForce(self,V,X):
        
        mod_V = np.linalg.norm(V)
        p = self.get_distance(X)
        cos_theta = self.get_cosine_angle(V,X)
        grad_P = self.grad_P(X)
        grad_cos = self.grad_cosine_angle(V,X)
        
        ## calculation is done in two parts. one is the sclar part and the other is the vector part.
        scalar_term = self.lambda_ * np.power(-cos_theta,self.beta -1) * mod_V/p
        
        
        vector_term = self.beta * grad_cos - ((grad_P/p[:,np.newaxis]).T * cos_theta).T
        phi_x_v = (vector_term.T * scalar_term ).T
        
        return phi_x_v
    
    
    def obstacle_force(self,V,X):
        
        mod_V = np.linalg.norm(V)        
        if mod_V == 0:
            return 0
        cosine_angle = self.get_cosine_angle(V,X)
        angle = np.arccos(cosine_angle)
        apply_force = angle > np.pi/2
        # apply_force = angle > np.pi/2 and angle <= np.pi
        if not np.array(apply_force).any():
            return 0
        
        else:
            force = self.calculateForce(V,X)
            calculatedForce = force[apply_force].copy()
            
            
        return calculatedForce.sum(axis = 0)
        
        # if apply_force:
        #     # define the variables    
        #     p = self.get_distance(X)
        #     cos_theta = self.get_cosine_angle(V,X)
        #     grad_P = self.grad_P(X)
        #     grad_cos = self.grad_cosine_angle(V,X)
            
        #     ## calculation is done in two parts. one is the sclar part and the other is the vector part.
        #     scalar_term = self.lambda_ * np.power(-cos_theta,self.beta -1) * mod_V/p
            
        #     vector_term = self.beta * grad_cos - cos_theta * grad_P/p
        #     phi_x_v = scalar_term * vector_term
            
        #     return phi_x_v
        
        # else:
        #     return np.zeros(self.n_dim)
        
    def step(self,dt):
        self.obstaclePos.append(self.currentPos)
        self.currentPos = self.currentPos + self.currentVel * dt       
          
if __name__ == "__main__":
    
    test__ = 1
    if test__ == 0:
        o1 = Obstacle(initPos = np.array([1,1]) , n_dim=2) ## 0,0
        v = np.array([1,1]) ## velocity of the end effector
        x = np.array([0.9,0.9]) ## position of the end effector
        
        r_x = x - o1.currentPos
        obstacle_force = o1.obstacle_force(v,r_x)
        print(obstacle_force)      
    
    if test__ == 1:
        
        obsPos = np.array([
            [1,0],
            [1,3],
        ])
        
        v = np.array([1,-1])
        x = np.array([0,1])
        r_x = x - obsPos
        obstacles = Obstacle(numObs = 2 , n_dim = 2 , initPos = obsPos)
        
        force = obstacles.obstacle_force(-v,r_x)
        print("forcee",force)
        
        # force = obstacles.obstacle_force(v,r_x)
        # print(force)
        