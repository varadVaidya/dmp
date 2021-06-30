import numpy as np
import sys,os

sys.path.append( sys.path[0] +'/..')
#print(sys.path)
class Gaussian():
    
    def __init__(self,width,center,weight = 1.0):
        
        self.width = width
        self.center = center
        self.weight = weight
    
    def evaluate(self,x):        
        return np.exp(-self.width * (x - self.center) * (x -self.center) )
    def weighted_evaluate(self,x):
        return self.weight * self.evaluate(x)
    
def plotGaussians(title,gaussian_values,numBasis,timearray):
    
    print("Plotting:" ,title)
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.title(title)
    
    for i in range(numBasis):
        plt.plot(timearray,gaussian_values[:,i])
    
    plt.show()
    

class functionApprox():
    
    ## assuming that we have forcing function as numpy array...
    
    @staticmethod
    def create_Gaussian_basis(numBasis,firstorderSystem,ifPlot=True):
        
        ## init the basis functions 
        center,width = np.empty(numBasis),np.empty(numBasis)
        
        PSI = numBasis * [None]
        
        des_activation = np.linspace(0,firstorderSystem.totaltime,numBasis)
        
        # ## calculate the centers
        
        center = np.exp(-firstorderSystem.alpha * np.linspace(0,1,numBasis) * firstorderSystem.totaltime)
        width = 1/np.gradient(center)**2
        
        # for i in range(numBasis):
        #     center[i] = np.exp(-firstorderSystem.alpha * des_activation[i])
        #     #center[i] = des_activation[i]
        #     #center[i] = np.exp( -firstorderSystem.alpha*des_activation[i]*(i)/(numBasis))
        #     #width[i] = numBasis / center[i] / firstorderSystem.alpha ## this width is final
        #center = np.linspace(0,firstorderSystem.totaltime,numBasis)
        ## calculate width
        # for j in range(numBasis-1):
        #     width[j] = 1/ ( (center[j+1] - center[j]) * ((center[j+1] - center[j])) ) 
        
        # width[-1] = width[-2]
        
        
        # for i in range(numBasis):
        #     width[i] = numBasis/center[i]**1.5/ firstorderSystem.alpha
        # width[0] = width[1]               
        print("centers",center)
        print("width",width)
        
        ## fill the gaussina class        
        gaussian_values = numBasis * [None]
        xvalues = firstorderSystem.simulation()
        for i in range(numBasis):
            
            PSI[i] = Gaussian(width[i], center[i])
            gaussian_values[i] = PSI[i].evaluate(xvalues)
        
        gaussian_values = np.array(gaussian_values).T
        if ifPlot:
            plotGaussians('Gaussians',gaussian_values,numBasis,xvalues)
            plotGaussians('Time Series Gaussian',gaussian_values,numBasis,firstorderSystem.timearray)
        
        return PSI,gaussian_values
        
    def generateRandomForcingFunc(numBasis,firstorderSystem):
        '''
        ? for testing of the method functionApprox::get_weights as the results are not accurate.
        '''
        PSI,values,weight_values = numBasis*[None], numBasis*[None], numBasis*[None]
        
        centers,widths,weights =  np.linspace(0,firstorderSystem.totaltime,numBasis),np.random.randint(1,3,size=(numBasis)) / 4 ,np.random.rand(numBasis)
        
        for i in range(numBasis):
            PSI[i] = Gaussian(widths[i], centers[i],weights[i])
            values[i] = PSI[i].evaluate(firstorderSystem.timearray)
            weight_values[i] = PSI[i].weighted_evaluate(firstorderSystem.timearray)
        
        values = np.array(values).T
        weight_values = np.array(weight_values).T
        print("shape weight_values:", np.shape(weight_values))
        weight_gaussian = np.sum(weight_values,axis=1)
        
        return weight_gaussian
        
    def get_weights(forcing_function,PSI,gaussian_values,firstorderSystem,numBasis):
        
        ## method 1 from the review paper. doeent work at all.
        ## causes alot of approximation errors in the computation.
        if len(forcing_function) != len(firstorderSystem.timearray):
            raise ValueError("forcing function must have the same dimensions as the time array.")
        
        xvalues = firstorderSystem.simulation()
        PSI_matrix = np.empty(shape=(len(xvalues) , numBasis))
            
        for i in range(numBasis):
            PSI_matrix[:,i] = PSI[i].evaluate(xvalues) *xvalues / np.sum(gaussian_values,axis=1)
        
        # for xval in range((len(xvalues))):
            
        #     denomSum = 0
            
        #     for N in range(numBasis):
        #         PSI_matrix[xval,N] = PSI[N].evaluate(xvalues[xval]) * xvalues[xval]
        #         denomSum = denomSum + PSI[N].evaluate(xvalues[xval])
            
        #     PSI_matrix[xval] = PSI_matrix[xval]/ denomSum
                
                
        
        weights = np.linalg.pinv(PSI_matrix).dot(forcing_function)
        
        #weights = np.dot(PSI_matrix.T, forcing_function)
        
        #print(weights)
        return weights   
    
    def get_weights_lstsquare(forcing_function,PSI,gaussian_values,firstorderSystem,numBasis):
        xvalues = firstorderSystem.simulation()
        PSI_matrix = np.empty(shape=(len(xvalues) , numBasis))
            
        # for i in range(numBasis):
        #     PSI_matrix[:,i] = PSI[i].evaluate(xvalues) *xvalues / np.sum(gaussian_values,axis=1)
        
        # for xval in range((len(xvalues))):
            
        #     denomSum = 0
            
        #     for N in range(numBasis):
        #         PSI_matrix[xval,N] = PSI[N].evaluate(xvalues[xval]) * xvalues[xval]
        #         denomSum = denomSum + PSI[N].evaluate(xvalues[xval])
            
        #     PSI_matrix[xval] = PSI_matrix[xval]/ denomSum
        
            
        weights = np.linalg.lstsq(PSI_matrix,forcing_function,rcond=None)[0]
        
        return weights
        
    def generateFrontTerm(firstorderSystem,secondorderSystem):
        xvalues = firstorderSystem.simulation()
        spatialScale = secondorderSystem.goal - secondorderSystem.x0[1]
        print(spatialScale)
        frontTerm = xvalues * spatialScale
        
        return frontTerm
        
    def get_weightsLWR(PSI,forcingFunc,numBasis,firstorderSystem,secondorderSystem):
        
        frontTerm = functionApprox.generateFrontTerm(firstorderSystem,secondorderSystem)
        weights = numBasis * [None]
        for i in range(numBasis):
            
            diagEntry = PSI[i].evaluate(firstorderSystem.simulation())
            # num = np.linalg.multi_dot((
            #     frontTerm.T, np.diag(diagEntry) , forcingFunc 
            # ))
            # den = np.linalg.multi_dot((
            #     frontTerm **2 , diagEntry
            # ))
            
            num = np.sum(frontTerm * np.diag(diagEntry), forcingFunc)
            den = np.sum(frontTerm **2 * np.diag(diagEntry))
            
            weights[i] = num/den
        
        return weights
        
if __name__ == '__main__':
    
    from dynamical_systems.canonical import FirstOrderDynamicalSystems
    from dynamical_systems.canonical import SecondOrderDynamicalSystem
    
    totaltime = 5
    phaseSystem = FirstOrderDynamicalSystems(alpha = 1,totaltime= totaltime)
    dynamicalSystem = SecondOrderDynamicalSystem(alpha =1,goal = 2,totaltime= totaltime)
    xvalues = phaseSystem.simulation()
    print("xvalues \n",xvalues)
    numBasis = 10
    forcingfunc = functionApprox.generateRandomForcingFunc(numBasis,phaseSystem)
    # forcingfunc[0:1000] ,forcingfunc[-1000:-1] = 0,0
    # forcingfunc[-1] = 0
    if len(phaseSystem.simulation()) != len(forcingfunc):
        print("fucked")
    
    PSI,gaussian_values = functionApprox.create_Gaussian_basis(numBasis,phaseSystem,ifPlot = True)
    
    
    if len(forcingfunc) != len(phaseSystem.timearray):
        raise ValueError("forcing func size invalid")
    print("stagw 1")
    #weights = functionApprox.get_weights(forcingfunc,PSI,gaussian_values,phaseSystem,numBasis)
    #weights = functionApprox.get_weightsLWR(PSI,forcingfunc,numBasis,phaseSystem,dynamicalSystem)\
    weights = functionApprox.get_weights_lstsquare(forcingfunc,PSI,gaussian_values,phaseSystem,numBasis)
    print("weights shape",weights.shape)
    print("weights",weights)
    print("stage 2")
    weighted_evaluate = [None] * numBasis
    for i in range(numBasis):
        PSI[i].weight = weights[i]
        weighted_evaluate[i] = PSI[i].weighted_evaluate(xvalues)
    
    print("satgw 2")
    weighted_evaluate = np.array(weighted_evaluate).T
    approximatedFunc = np.sum(weighted_evaluate,axis=1)
    
    import matplotlib.pyplot as plt
    plt.plot(phaseSystem.timearray,forcingfunc,label = "forcing func")
    plt.plot(phaseSystem.timearray,approximatedFunc,label="approximatedFunc")
    plt.title('Approximate Functions')
    plt.legend()
    plt.show()
    plotGaussians('Function Approximate',weighted_evaluate,numBasis,phaseSystem.timearray)
    
            
        
        