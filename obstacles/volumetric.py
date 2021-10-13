import numpy as np

## class to contain the method of volumetric obstacle avoidance

class VolumetricObstacle():
    
    ##contains the parameters of the of volumetric obstacle.
    
    def __init__(self,center,axes,n_dim = 3,lambda_ = 1,beta = 2 , eta = 0.5):
        
        self.n_dim = n_dim
        
        self.center = center
        self.axes = axes
        
        ## raise value error if the len of center and axes is not same
        if len(center) != len(axes):
            raise ValueError("center and axes must be same dimension")
        
        if len(axes) != n_dim:
            raise ValueError("center and axes are not the same dimension as the obstacle dimension")
        
        """
        if initPos is None then assign both the initial position and velocity to be zero of array size n_dim
        if initPos is not None then assign the initial position and velocity to be the input values
        """
        ### !!!!! INIT POS AND VELOCIY are not used in the code !!!!! will be replaced with center and axes.
        
        self.initPos = center
        self.initVel = np.zeros(n_dim)
        
        self.currentPos = self.initPos.copy() ## current position of the obstacle
        self.currentVel = self.initVel.copy() ## current velocity of the obstacle
        
        
        self.lambda_ = lambda_
        self.beta = beta
        self.eta = eta
    
    def calc_C_x(self,X):
        """
        calculate the isopotential of the obstacle at any given point by formula:
        
        ((x1-center1)/2)^2 + ((x2-center2)/l2)^2 + ((x3 - center3)/l3)^2
        
        this is also named by C(x) in the paper
        """
        return np.sum(np.power((X-self.center)/self.axes,2))
    
    def get_distance(self,X):
        """
        return the norm of the X vector
        """
        return np.linalg.norm(X)
    
    def grad_Cx(self,X):
        """
        returns the gradients of the iso potential field given by:
        
        gradISO = 2 * (X - center)/axes**2
        """
        return 2*(X-self.center)/self.axes**2
    
    def abs_grad_C_x(self,X):
        """
        return the absolute value of the norm of the gradient of the iso potential field where gradient is calculated by gradisoPotential
        """
        return np.linalg.norm(self.grad_Cx(X))
    
    def grad_grad_Cx_dotV(self,X,V):
        """
        returns the gradient of the dot product of the gradient of the iso potential field and velocity given by:
        
        grad_gradISOvel = 2 * (V/axes**2)
        """
        return 2*(V/self.axes**2)
    
    def grad_abs_grad_C_x(self,X):
        """
        returns the gradient of the absolute value of the gradient of isopotential function given by the formula
        
        grad_abs_gradisoPotential = (4/abs_gradisoPotential) * (X - center)/axes**4
        """
        return ( 4/self.abs_grad_C_x(X) ) * ( X-self.center )/self.axes**4
    
    def cos_theta(self,X,V):
        """
        returns the cosine of the angle between the velocity and the gradient of the iso potential field. clip the value of it between -1 and 1.
        """
        return np.clip(np.dot(self.grad_Cx(X),V)/(np.linalg.norm(V)*self.abs_grad_C_x(X)),-1,1)
    
    def grad_cos_theta(self,X,V):
        """
        returns the gradient of the cosine of the angle between the velocity and the gradient of the iso potential field
        
        the gradient is a product of scalar term and a vector term.
        
        scalarTerm = 1/( abs(V) * C_x ** 2 )
        
        
        vectorTerm = abs(grad_C_x) * grad_grad_Cx_dotV - (grad_Cx dot V) * grad_abs_grad_C_x
        """
        scalarTerm = 1/( np.linalg.norm(V) * self.calc_C_x(X)**2 )
        
        vectorTerm = self.abs_grad_C_x(X) * self.grad_grad_Cx_dotV(X,V) - np.dot(self.grad_Cx(X),V) * self.grad_abs_grad_C_x(X)
        
        return scalarTerm * vectorTerm
    
    def obstacle_force(self,V,X):
        
        mod_V = np.linalg.norm(V)
        
        if mod_V == 0:
            return 0
        
        cos_theta = self.cos_theta(X,V)
        # print(cos_theta,": costheta")
        angle = np.arccos( self.cos_theta(X,V) )
        # print(angle)
        apply_force = angle > np.pi/2 and angle <= np.pi
        
        if apply_force:
            
            C_x = self.calc_C_x(X)
            cos_theta = self.cos_theta(X,V)
            grad_cos_theta = self.grad_cos_theta(X,V)
            grad_Cx = self.grad_Cx(X)
            
            scalarTerm = -self.lambda_ * mod_V * np.power(-cos_theta,self.beta -1 ) / np.power(C_x,self.eta)
            
            vectorTerm = -self.beta * grad_cos_theta + (self.eta * cos_theta / C_x) * grad_Cx
            
            phi_x_v = scalarTerm * vectorTerm
            
            return phi_x_v
        
        else:
            return 0
    
    def step(self,dt):
        pass

if __name__ == "__main__":
    
    volObs = VolumetricObstacle(center =  np.array([0,0,0]),
                                axes = np.array([1,1,1]))
    
    
    V = np.array([1.1,1.1,1.1])
    X = 4 * np.array([1,1,1])
    r_x = r_x = X - volObs.center
    
    force = volObs.obstacle_force(-V,r_x)
    
    print(force)
    
            
            
        
        
        
        
    
        
        