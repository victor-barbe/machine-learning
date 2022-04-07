import numpy as np
import matplotlib.pyplot as plt
import random 
import math

#importing data
data = np.loadtxt("data_kmeans.txt", dtype=float)
#our centroids
centroids = np.zeros((3,2))
#our distance vector
distance = np.zeros((len(data),3)) 
#our output vector
Y = np.zeros((len(data),1))
#indicator
I = np.zeros((len(data),1))


for i in range (3):
    centroids[i][0] = random.uniform(0,9)  #Define X coordinate
    centroids[i][1] = random.uniform(0,6)     #define Y coordinate

#ploting the training data to get an idea
def plotTrainingData():
    plt.title("Training data")  
    plt.xlabel("X1")
    plt.ylabel("X2")
    x = [column[0] for column in data]
    y = [column[1] for column in data]
    z = [column[0] for column in centroids]
    u = [column[1] for column in centroids]
    plt.scatter(x, y, s = 10)
    plt.scatter(z, u, c='lightblue')
    plt.show()


def indicatorFunction():
    return 0

def kmeanAlgorithm():
    condition = True
    while condition == True:
        sumCentroid0 =  np.zeros((1,2))
        sumCentroid1 =  np.zeros((1,2))
        sumCentroid2 =  np.zeros((1,2))
        counterC0 = 0
        counterC1 = 0
        counterC2 = 0
        for i in range(len(data)):
            for j in range(3):
                distance[i][j] = np.sqrt(np.power(data[i][0] - centroids[j][0],2) + np.power(data[i][1] - centroids[j][1],2) )
            Y[i] = np.argmin(distance[i])
            #condition function
            if Y[i] == 0: #if the point is in centroid 1
                sumCentroid0[0][0] += data[i][0]
                sumCentroid0[0][1] += data[i][1]
                counterC0 +=1
                
            elif Y[i] == 1:
                sumCentroid1[0][0] += data[i][0]
                sumCentroid1[0][1] += data[i][1]
                counterC1 +=1
            elif Y[i] == 2:
                sumCentroid2[0][0] += data[i][0]
                sumCentroid2[0][1] += data[i][1]
                counterC2 +=1 

            #for coordinate (X)
            centroids[0][0] = sumCentroid0[0][0] / counterC0 
            #for coordinate (Y)
            centroids[0][1] = sumCentroid0[0][1] / counterC0 

            #for coordinate (X)
            centroids[1][0] = sumCentroid1[0][0] / counterC1 
            #for coordinate (Y)
            centroids[1][1] = sumCentroid1[0][1] / counterC1 

            #for coordinate (X)
            centroids[2][0] = sumCentroid2[0][0] / counterC2 
            #for coordinate (Y)
            centroids[2][1] = sumCentroid2[0][1] / counterC2             
    return 0


print(distance)

plotTrainingData()
kmeanAlgorithm()