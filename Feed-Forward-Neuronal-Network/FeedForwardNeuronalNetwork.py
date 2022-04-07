import numpy as np
import math
import matplotlib.pyplot as plt

data = np.loadtxt("data_ffnn_3classes.txt", dtype=float)

N = 2 #number of feature
I = 71 #number of samples
K = 2 #number of hidden neurons
J = 3 #number of predicted classes
alpha_v = 0.01 #learning rates
alpha_w = 0.01 

#we create random values for the weights, here we have 3 for the 3 values of each feature and K for K neurons
V = np.random.rand(N+1, K)
#we create random weights for the second layer of neurons with 5 neurons
W = np.random.rand(K+1, J)

#function sigmoid to compute activation function
def sigmoid(x):
    return  1 / (1 + np.exp(-x))

iteration = 0

#we delete the predicted class
X = np.delete(data, 2, axis=1)

#we create an array of one to create the bias value
B = np.ones((71,1))

#We create X_ by adding the bias value
X_ = np.concatenate((B,X), axis =1)

#question 1 plot the training data
def plotTrainingData():
    plt.title("Training data")
    plt.xlabel("X1")
    plt.ylabel("X2")
    x = [column[0] for column in X]
    y = [column[1] for column in X]
    colors = [column[2] for column in data]
    plt.scatter(x, y, s = 10, c = colors)
    plt.show()

#plotTrainingData()

errorArray = []
numberOfLoops = 100

#training
while iteration < numberOfLoops:

    #now we multiply the value of the input with the weight.
    X__ = np.dot(X_, V)

    #print(X__)
    #Here we apply the activation function of the first layer
    F = sigmoid(X__)

    F = np.reciprocal(F)

    #Here we add the bias for the second layer of neurons
    F_ = np.concatenate((B,F), axis =1)

    #We multiply the input that came from the first layer with the weight of the second layer
    F__ = np.dot(F_, W) 
    #we apply the second activiation function (the same in this case)

    G = sigmoid(F__)

    #normalize the vector Y to compute error
    Y = np.zeros((71, 3))

    for i in range(71):
        if (data[i][2] == 0):
            Y[i][0] = 1
        elif (data[i][2] == 1):
            Y[i][1] = 1
        elif (data[i][2] == 2):
            Y[i][2] = 1

    E = 0
    for i in range(len(G)):
        for j in range(len(G[0])):
                E += math.pow(Y[i][2] - G[i][j], 2)
    E = 0.5 * E
    
    errorArray.append(E)

    #print("The error at the iteration number ",iteration, "is : ", E)
    #print("          ")

    #K ajouter 1 ? problème bias on ne veut pas prendre F0 dans le calcul de W pour la seconde ligne

    derivate = 0
    #update The Weight W 
    for k in range(K+1):
        for j in range(J):
            for i in range(I):
                derivate = derivate + (G[i][j] - Y[i][j]) * G[i][j] * (1 - G[i][j]) * F_[i][k]
            W[k][j] = W[k][j] - alpha_w * derivate
            derivate = 0

    #update The Weight V
    for n in range(N):
        for k in range(K):
            for i in range(I):
                for j in range(J):
                    derivate = (G[i][j] - Y[i][j]) * G[i][j] * ( 1 - G[i][j]) * W[k+1][j] * F[i][k] * (1 - F[i][k]) * X_[i][n]
            V[n][k] = V[n][k] - alpha_v * derivate
            derivate = 0
    
    iteration = iteration + 1


print("Optimized weight of layer V at iteration", iteration, "are : ")
for i in range(len(W)):
    for j in range (len(W[0])):
        print(W[i][j])

print("Optimized weight of layer W at iteration", iteration, "are : ")
for i in range(len(V)):
    for j in range (len(V[0])):
        print(V[i][j])

#print(G)
#print(errorArray)

def PlotError(errors):
    plt.title("Error over training")
    plt.xlabel("X1")
    iteration = [i for i in range(numberOfLoops)]
    plt.plot(iteration, errors)
    plt.show()

#PlotError(errorArray)

#compare the predicted result and the actual result for the dataset



def predictedResultsComparaison(input, output):
    incorrectPredictions = 0
    correctPredictions = 0
    for i in range (I):
        #print("The actual class was :", input[i][2], "and the neuronal network gave : ")
        #print("Probability equal to class 0", output[i][0])
        #print("Probability equal to class 1", output[i][1])
        #print("Probability equal to class 2", output[i][2])
        print("  ")

        if output[i][0] > output[i][1] and output[i][0] > output[i][2] and input[i][2] == 0:
            print("S 0 was predicted correctly with a probability of : ", output[i][0])
            correctPredictions = correctPredictions + 1

        elif output[i][1] > output[i][0] and output[i][1] > output[i][2] and input[i][2] == 1:
            print("Class 1 was predicted correctly with a probability of : ", output[i][1])
            correctPredictions = correctPredictions + 1

        elif output[i][2] > output[i][0] and output[i][2] > output[i][1] and input[i][2] == 2:
            print("Class 2 was predicted correctly with a probability of : ", output[i][2])
            correctPredictions = correctPredictions + 1

        else:
            print("The class which was: ", input[i][2]," wasn't predicted correctly")
            incorrectPredictions = incorrectPredictions + 1

    print("In the end over ", I, " predictions, there were ", correctPredictions, " correct predictions and ", incorrectPredictions, " incorrect predictions")

#predictedResultsComparaison(data, G)

#in this last part we will try to test our program

dataTest = np.loadtxt("data2.txt", dtype=float)
x = np.delete(dataTest, 2, axis=1)


def TestNeuronNetwork():
    b = np.ones((3,1))

    x_ = np.concatenate((b,x), axis =1)

    x__ = np.dot(x_, V)
 
    #Here we apply the activation function of the first layer
    
    f = sigmoid(x__)
    f = np.reciprocal(f)
    
    #Here we add the bias for the second layer of neurons
    f_ = np.concatenate((b,f), axis =1)

    #We multiply the input that came from the first layer with the weight of the second layer
    f__ = np.dot(f_, W) 
    #we apply the second activiation function (the same in this case)

    g = sigmoid(f__)

    y = []
    for i in range(3):
        if g[i][0] > g[i][1] and g[i][0] > g[i][2]:
            y.append(0)
        elif g[i][1] > g[i][0] and g[i][1] > g[i][2]:
            y.append(1)
        elif g[i][2] > g[i][0] and g[i][2] > g[i][1]:
            y.append(1)



    return(g,y)

gTest = TestNeuronNetwork()[0]
yTest = TestNeuronNetwork()[1] 
print(yTest)

def plotTestData():
    plt.title("Test data")
    plt.xlabel("X1")
    plt.ylabel("X2")
    xTest = [column[0] for column in x]
    yTest = [column[1] for column in x]
    colors =  yTest
    plt.scatter(xTest, yTest, s = 10, c = colors)
    plt.show()

plotTestData()