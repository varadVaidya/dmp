import numpy as np

class Gaussian():
    
    ## plan is to make a class that will make a gaussion function as a basis function for the dynamic motion primitives
    
    def __init__(self,h,c,weight = 1.0):
        """ init the Gaussian class \n

        Args: \n
            h (float): width od the Gaussian \n
            c (float): center of the gaussian \n
            weight (float): the weight of as particular child of the guassian to be used in function approximation. Defaults to 1.0 .\n
            
        """
        self.h = h
        self.c = c
        self.weight = weight
        
    def evaluate(self,x):
        """
        evaluate the Gaussian at a particular x value. or at an array of values. \n
        

        Args:\n
            x (float or np 1D array): arrayof float for which the Gaussian is to be evaluated \n

        Returns: \n
            [np array 1D]: Gaussian value at the given x. \n
        """
        
        ## evaluate the Gaussian at a particular x value.
        ## x value could be an numpy 1D array as well.
        
        return np.exp(-self.h * (x - self.c) * (x - self.c) )
    
    def weighted_evaluate(self, x):
        ## placeholder function to calculate weighted gaussian distribution
        ## will just multiply the weight to the normal evaluation.
        return self.weight * self.evaluate(x)
    
if __name__ == '__main__':
    
    
    noBasis = 10
    psi = noBasis * [None]
    values = noBasis * [None]
    weight_values = noBasis * [None]
    
    
    xvalues = np.linspace(-10,10,200)
    
    centers,widths,weights =  np.arange(-5,5),np.random.randint(1,3,size=(noBasis))/3,np.random.rand(noBasis)
    for i in range(noBasis):
        
        psi[i] = Gaussian(widths[i],centers[i],weights[i])
        values[i] = psi[i].evaluate(xvalues)
        weight_values[i] = psi[i].weighted_evaluate(xvalues)
            
    
    
    #print(x)
    values = np.array(values).T
    weight_values = np.array(weight_values).T
    print(values.shape)
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.title('Basis functions')
    
    for i in range(noBasis):
        plt.plot(xvalues,values[:,i])
        
        
    plt.figure()
    plt.title('weighted basis functions')
    
    for i in range(noBasis):
        plt.plot(xvalues,weight_values[:,i])
        plt.bar(centers[i],weights[i],width = 0.2)
    
        
    weight_gaussian = np.sum(weight_values,axis=1)
    plt.plot(xvalues,weight_gaussian ,label = 'weight_gaussian')
    plt.legend()
    plt.show()
        
    


