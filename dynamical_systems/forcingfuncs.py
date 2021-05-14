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


def get_weights_from_forcing_functions(forcing_funcs):
    
    """
    //TODO:
    CONVERT THE WEIGHT CALCULATION DONE BELO TO A function
    Need to have higher priority.
    """
    pass
    
    
     
if __name__ == '__main__':
    
    
    noBasis = 25
    psi = noBasis * [None]
    values = noBasis * [None]
    weight_values = noBasis * [None]
    
    
    xvalues = np.linspace(-15,15,2000)
    
    centers,widths,weights =  np.arange(-12,13),np.random.randint(1,3,size=(noBasis)) / 4 ,np.random.rand(noBasis)
    
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
    
    
    ## for some time we will assume that the forcing function is basically the weighted gaussian to check if the method that we are using works properly
    ## we should get the weightd that we applied back.
    
    forcing_function = weight_gaussian
    
    PSI_matrix = np.empty(shape=(len(xvalues) , noBasis))
    
    for i in range(noBasis):
        
        PSI_matrix[:,i] = psi[i].evaluate(xvalues) * xvalues / np.sum(values,axis=1)
    calculated_weights = np.linalg.pinv(PSI_matrix).dot(forcing_function)
    print(calculated_weights)
    print(weights)
    
    calc_psi = noBasis * [None]
    calc_values = noBasis * [None]
    calculated_weight_values = noBasis * [None]
    for i in range(noBasis):
        
        calc_psi[i] = Gaussian(widths[i],centers[i],calculated_weights[i])
        calc_values[i] = calc_psi[i].evaluate(xvalues)
        calculated_weight_values[i] = calc_psi[i].weighted_evaluate(xvalues)
    
    calc_values = np.array(calc_values).T
    calculated_weight_values = np.array(calculated_weight_values).T
    calc_weight_gaussian = np.sum(calculated_weight_values,axis=1)
    
    plt.plot(xvalues,weight_gaussian ,'g|',label = 'calc_weight_gaussian')
    plt.legend()
    plt.show()