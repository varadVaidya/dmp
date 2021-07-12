## point obstacle container class
## notations used here are from the paper refered in the README
import numpy as np
class Obstacle():
    ## will contain all the obstacle info necessary:
    ## also has the paramerters to define the dynamic potential field
    ## obstacle can also move with time
    def __init__(self,n_dim = 3,initPos = None ,initVel = None):
        
        self.n_dim = n_dim ## dimension in which the obstacle exists
        
        if initPos is None:
            self.initPos = np.zeros(self.n_dim) ## the initial position of the obstacle
            #self.initVel = np.zeros(3) ## initial velocity of of the obstacle
        else:
            self.initPos = initPos
            #self.initVel = initVel
        
        self.currentPos = self.initPos.copy() ## current position of the obstacle
        #self.currentVel = self.initVel.copy() ## current velocity of the obstacle
        ## parameters for the dynamic potential field.
        self.speed = np.zeros(self.n_dim) ## speed of the obstacle.
        self.lambda_ = 6
        self.beta = 2
    
    def get_distance(self,X):
        ## distance between obstacle and position of the end effector
        distance = np.linalg.norm(X - self.currentPos)
        return distance
        
    def get_cosine_angle(self,V,X):
        ## get the angle between the velocity vector and the position vector of the end effector relative to the obstacle.
        numerator = np.dot(V,X)
        mod_V = np.linalg.norm(V) # modulus of the velocity vector
        
        denominator = mod_V * self.get_distance(X)
        
        cosine_angle = numerator/denominator
        
        return cosine_angle
    
    def grad_P(self,X):
        
        delta_P = (X - self.currentPos)/self.get_distance(X)
        
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
        
    
        if self.get_cosine_angle(V,X) <= 0:
            # define the variables    
            mod_V = np.linalg.norm(V)
            p = self.get_distance(X)
            cos_theta = self.get_cosine_angle(V,X)
            grad_P = self.grad_P(X)
            grad_cos = self.grad_cosine_angle(V,X)
            
            ## calculation is done in two parts. one is the sclar part and the other is the vector part.
            scalar_term = self.lambda_ * np.power(-cos_theta,self.beta -1) * mod_V/p
            
            vector_term = self.beta * grad_cos - cos_theta * grad_P/p
            phi_x_v = scalar_term * vector_term
            
            return phi_x_v
        
        else:
            return np.zeros(self.n_dim)
        
    def step(self,dt):
        self.currentPos += self.speed * dt
if __name__ == "__main__":
    
    #import numpy as np
    o1 = Obstacle(n_dim=2) ## 0,0
    v = np.array([1,1]) ## velocity of the end effector
    x = np.array([-1,-1]) ## position of the end effector
    
    obstacle_force = o1.obstacle_force(v,x)
    print(obstacle_force)      
        
        
        