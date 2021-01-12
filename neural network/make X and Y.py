import numpy as np
from scipy import optimize
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import csv
# X = np.array([[]], dtype=float)
# y = np.array([[]], dtype=float)
def getXY(first,second,third):
    data = pd.read_csv("feature.csv")
    X = np.array([[]], dtype=float)
    datarow = data.loc[first]
    # for j in range(0,1):
    #     datarow = data.loc[j]
    feature = []
    for i in range(2,4):
        a=datarow[i]
        feature.append(a)
    X=np.array(feature)

    for j in range(first,second):
        datarow = data.loc[j]
        feature = []
        for i in range(2,4):
            a = datarow[i]
            feature.append(a)
        b=np.array(feature)
        X = np.vstack((X, b))

    X/np.amax(X, axis=0)
    y = np.array([[]], dtype=float)
    datarow = data.loc[second]
    # for j in range(0,1):
    #     datarow = data.loc[j]
    feature = []
    a=datarow[10]
    feature.append(a)
    y=np.array(feature)

    for j in range(first,second):
        datarow = data.loc[j]
        feature = []

        a = datarow[10]
        feature.append(a)
        b=np.array(feature)
        y = np.vstack((y, b))

    y = y #Max test score is 100

    trainX =X
    trainY =y
    X = np.array([[]], dtype=float)
    datarow = data.loc[second]
    # for j in range(0,1):
    #     datarow = data.loc[j]
    feature = []
    for i in range(2,4):
        a=datarow[i]
        feature.append(a)
    X=np.array(feature)

    for j in range(second,third):
        datarow = data.loc[j]
        feature = []
        for i in range(2, 4):
            a = datarow[i]
            feature.append(a)
        b=np.array(feature)
        X = np.vstack((X, b))

    X/np.amax(X, axis=0)


    y = np.array([[]], dtype=float)
    datarow = data.loc[second]
    # for j in range(0,1):
    #     datarow = data.loc[j]
    feature = []
    a=datarow[10]
    feature.append(a)
    y=np.array(feature)

    for j in range(second,third):
        datarow = data.loc[j]
        feature = []

        a = datarow[10]
        feature.append(a)
        b=np.array(feature)
        y = np.vstack((y, b))

    y = y/10 #Max test score is 100
    testX =X
    testY =y
    return trainX,trainY,testX,testY

trainX,trainY,testX,testY=getXY(1,1000,1500)
# print(trainX,trainY,testX,testY)
# y = y/100 #Max test score is 100
X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

X = X/np.amax(X, axis=0)
y = y/10 #Max test score is 100
# #Training Data:
# trainX = np.array(([3,5], [5,1], [10,2], [6,1.5]), dtype=float)
# trainY = np.array(([75], [82], [93], [70]), dtype=float)
#
# #Testing Data:
# testX = np.array(([4, 5.5], [4.5,1], [9,2.5], [6, 2]), dtype=float)
# testY = np.array(([70], [89], [85], [75]), dtype=float)
#
# #Normalize:
# trainX = trainX/np.amax(trainX, axis=0)
# trainY = trainY/100 #Max test score is 100
#
# #Normalize by max of training data:
# testX = testX/np.amax(trainX, axis=0)
# testY = testY/100 #Max test score is 100


#Regularization Parameter:
Lambda = 0.0001


# New complete class, with changes:
class Neural_Network(object):
    def __init__(self, Lambda=0):
        # Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        # Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

        # Regularization Parameter:
        self.Lambda = Lambda

    def forward(self, X):
        # Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1 / (1 + np.exp(-z))

    def sigmoidPrime(self, z):
        # Gradient of sigmoid
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def costFunction(self, X, y):
        # Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5 * sum((y - self.yHat) ** 2) / X.shape[0] + (self.Lambda / 2) * (
        np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return J

    def costFunctionPrime(self, X, y):
        # Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
        # Add gradient of regularization term:
        dJdW2 = np.dot(self.a2.T, delta3) / X.shape[0] + self.Lambda * self.W2

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        # Add gradient of regularization term:
        dJdW1 = np.dot(X.T, delta2) / X.shape[0] + self.Lambda * self.W1

        return dJdW1, dJdW2

    # Helper functions for interacting with other methods/classes
    def getParams(self):
        # Get W1 and W2 Rolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        # Set W1 and W2 using single parameter vector:
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end],
                             (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end],
                             (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

def computeNumericalGradient(N, X, y):
    paramsInitial = N.getParams()
    numgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    e = 1e-4

    for p in range(len(paramsInitial)):
        # Set perturbation vector
        perturb[p] = e
        N.setParams(paramsInitial + perturb)
        loss2 = N.costFunction(X, y)

        N.setParams(paramsInitial - perturb)
        loss1 = N.costFunction(X, y)

        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2 * e)

        # Return the value we changed to zero:
        perturb[p] = 0

    # Return Params to original value:
    N.setParams(paramsInitial)

    return numgrad

##Need to modify trainer class a bit to check testing error during training:
class trainer(object):
    def __init__(self, N):
        # Make Local reference to network:
        self.N = N

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))
        self.testJ.append(self.N.costFunction(self.testX, self.testY))

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X, y)

        return cost, grad

    def train(self, trainX, trainY, testX, testY):
        # Make an internal variable for the callback function:
        self.X = trainX
        self.y = trainY

        self.testX = testX
        self.testY = testY

        # Make empty list to store training costs:
        self.J = []
        self.testJ = []

        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS',
                                 args=(trainX, trainY), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res

NN = Neural_Network(Lambda=0.0001)
#Make sure our gradients our correct after making changes:
numgrad = computeNumericalGradient(NN, X, y)
grad = NN.computeGradients(X,y)
#Should be less than 1e-8:
# print(np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad))
T = trainer(NN)
T.train(trainX, trainY, testX, testY)
plot.plot(T.J)
plot.plot(T.testJ)
plot.grid(1)
plot.xlabel('Iterations')
plot.ylabel('Cost')
# plot.show()
#Test network for various combinations of sleep/study:
hoursSleep = np.linspace(0, 10, 100)
hoursStudy = np.linspace(0, 5, 100)

#Normalize data (same way training data way normalized)
hoursSleepNorm = hoursSleep/10.
hoursStudyNorm = hoursStudy/5.

#Create 2-d versions of input for plotting
a, b  = np.meshgrid(hoursSleepNorm, hoursStudyNorm)

#Join into a single input matrix:
allInputs = np.zeros((a.size, 2))
allInputs[:, 0] = a.ravel()
allInputs[:, 1] = b.ravel()

allOutputs = NN.forward(allInputs)

#
# numgrad = computeNumericalGradient(NN,X,y)
# print(numgrad)
#

#
# print(NN.costFunctionPrime(X,y))
# print(NN.forward(X))
# print(y)

#Contour Plot:
yy = np.dot(hoursStudy.reshape(100,1), np.ones((1,100)))
xx = np.dot(hoursSleep.reshape(100,1), np.ones((1,100))).T

CS = plot.contour(xx,yy,10*allOutputs.reshape(100, 100))
plot.clabel(CS, inline=1, fontsize=10)
plot.xlabel('Volume')
plot.ylabel('DJ')
# plot.show()


#3D plot:
##Uncomment to plot out-of-notebook (you'll be able to rotate)
#%matplotlib qt
# fig = plot.figure()
# plot.ax = fig.gca(projection='3d')
#
# plot.ax.scatter(10*trainX[:,0], 10*trainX[:,1], 10*trainY, c='k', alpha = 1, s=30)
#
#
# surf = plot.ax.plot_surface(xx, yy, 10*allOutputs.reshape(100, 100),
#                        cmap=plot.cm.jet, alpha = 0.5)
# plot.ax.set_xlabel('Volume')
# plot.ax.set_ylabel('DJ')
# plot.ax.set_zlabel('level')
# plot.show()
yHat = NN.forward(testX)


with open('checkresult.csv', 'w') as f:
    f_csv = csv.writer(f)
    portfolio =[]
    for i in range(0, 150):
        Y=testY[i]
        y=Y[0]
        VHat = yHat[i]
        vhat =VHat[0]
        value = []
        value.append(y)
        value.append(vhat)
        f_csv.writerow(value)

# for i in range(0,)
# with open('checkresult.csv','w') as f:
#     f_csv = csv.writer(f)
#     for i in range(0,251):
#         f_csv.writerow(portfolio[i])
# print(value)