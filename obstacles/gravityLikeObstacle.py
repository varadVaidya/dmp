## volumetric obstacle container class
## notations used here are from the paper refered in the README
import numpy as np
class GravityLikeObstacle():
    ## will contain all the obstacle info necessary:
    ## also has the paramerters to define the dynamic potential field
    ## obstacle can also move with time
    def __init__(self,n_dim = 3,initPos = None ,initVel = None,lambda_ = 1,n = 2, r = 1):
        
        self.n_dim = n_dim ## dimension in which the obstacle exists
        
        if initPos is None:
            self.initPos = np.zeros(self.n_dim) ## the initial position of the obstacle
            self.initVel = np.zeros(self.n_dim) ## initial velocity of of the obstacle
            
        if initPos is not None:
            
            self.initPos = initPos
            
            if initVel is None:
                self.initVel = np.zeros(self.n_dim) ## the initial velocity of the obstacle
            
            else:
                self.initVel = initVel
        
        self.currentPos = self.initPos.copy() ## current position of the obstacle
        self.currentVel = self.initVel.copy() ## current velocity of the obstacle
        ## parameters for the dynamic potential field.
        
        self.lambda_ = lambda_ ## the strength of the potential field 
        self.beta = 2
        self.n = n ## the order of the potential field
        self.r = r ## the radius of the obstacle
        
        
        
        self.obstaclePos = []
        
        
    
    def get_distance(self,X):
        ## distance between obstacle and position of the end effector
        distance = np.linalg.norm(X)
        return distance
        
    def get_cosine_angle(self,V,X):
        ## get the angle between the velocity vector and the position vector of the end effector relative to the obstacle.
        numerator = np.dot(V,X)
        mod_V = np.linalg.norm(V) # modulus of the velocity vector
        
        denominator = mod_V * self.get_distance(X)
        
        cosine_angle = numerator/denominator
        
        return cosine_angle
    
    def grad_P(self,X):
        
        delta_P = (X)/self.get_distance(X)
        
        return delta_P
    
    def grad_cosine_angle(self,V,X):
        
        distance = self.get_distance(X)
        mod_V = np.linalg.norm(V)
        vTx = V.dot(X)
        delta_P = self.grad_P(X)
        
        numerator = distance * V - vTx * delta_P
        
        denominator = mod_V * distance **2
        
        delta_cos = numerator/denominator
        
        return delta_cos

    def obstacle_force(self,V,X):
        
        mod_V = np.linalg.norm(V)
        
        if mod_V == 0:
            return 0

        angle = np.arccos(self.get_cosine_angle(V,X))
        
        apply_force = angle > np.pi/2 and angle <= np.pi
        
        if apply_force:
            # define the variables    
            p = self.get_distance(X)
            p_raise_n = np.power(p,self.n)
            cos_theta = self.get_cosine_angle(V,X)
            grad_P = self.grad_P(X)
            grad_cos = self.grad_cosine_angle(V,X)
            
            ## calculation is done in two parts. one is the sclar part and the other is the vector part.
            
            scalar_term = self.lambda_ * np.power(-cos_theta,self.beta -1) * mod_V/p_raise_n
            
            vector_term = self.beta * grad_cos - self.n * cos_theta * grad_P/p
            phi_x_v = scalar_term * vector_term
            
            return phi_x_v
        
        else:
            return np.zeros(self.n_dim)
        
    def step(self,dt):
        self.obstaclePos.append(self.currentPos)
        self.currentPos = self.currentPos + self.currentVel * dt
