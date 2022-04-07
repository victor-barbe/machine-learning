import numpy as np
import random

#defining the possible state, all minus the obstacle
states = []
for i in range(3):
    for j in range(4):
            states.append((i,j))

#we delete the obstacle in our grid            
states.pop(5)

#defining our reward, hell heaven and -0.02 the rest of the time
rewards = {}
for i in states:
    if i == (0,3):
        rewards[i] = 1
    elif i == (1,3):
        rewards[i] = -1
    else:
        rewards[i] = -0.02
    
#all the possible actions, taking into account obstacle
generalActions = {
    (0,0):('S', 'E'), 
    (0,1):('E', 'W'),    
    (0,2):('S', 'W', 'E'),
    (1,0):('S', 'N'),
    (1,2):('S', 'E', 'N'),
    (2,0):('N', 'E'),
    (2,1):('W', 'E'),
    (2,2):('W', 'E', 'N'),
    (2,3):('W', 'N'),
    }

#initialize our policy randomly
PI = {}
for s in generalActions.keys():
    PI[s] = np.random.choice(generalActions[s])

#print(PI)

#initialize our value function
V = {}
for s in states:
    #we initialize it at 0 for all the grid
    if s in generalActions.keys():
        V[s] = 0
    #we initialize it at 1 for heaven
    if s == (0,3):
        V[s]= 1
    #we initialize it at -1 for hell
    if s == (1,3):
        V[s]= -1

#We initialize Q with a value for each state, and a random action
tuple = {'N' : 0, 'S' : 1, 'E' : 2, 'W' : 3}
Q = {}
for s in states:
    for a in generalActions.keys():
        #we initialize it at 0 for all the grid
        if s in generalActions.keys():
            #Q[s] = ['N', 0, 'S', 1, 'W', 0, 'E', 1]
            Q[s] = tuple
        #we initialize it at 1 for heaven
        if s == (0,3):
            #Q[s]= [1, np.random.choice(generalActions[a]), 0 , 'N', 1, 'S']
            Q[s] = tuple

        #we initialize it at -1 for hell
        if s == (1,3):
            #Q[s]= [-1, np.random.choice(generalActions[a]) ,0 , 'N', 1, 'S'] 
            Q[s] = tuple


#value iteration algorithm
def valueIterationAlgorithm(gamma = 0.9, convergenceCondition = 0.005):
    convergence = True
    while convergence == True:
        variation = 0
        for s in states:   #we take all the states
            if s in PI:    #we don't take the heaven and hell were movement doesn't happen
            
                old_v = V[s]   #old value of V(s) we need to check if we want to stop
                new_v = 0      #new value of V(S) 
                
                #nextState will be the next position in the grid we want
                #randomAct1 and randomAct2 will be the random action we might get 
                for a in generalActions[s]:
                    if a == 'N':
                        nextState = [s[0]-1, s[1]]

                        #if randomMove to the left of North (West)
                        if s[0] == 1 and s[1] == 2 or s[1] == 0: #not possible when obstacle or end of grid
                            randomAct1 = [s[0], s[1]]
                        else:
                            randomAct1 = [s[0], s[1]-1]

                        #if randomMove to the right of North (east)
                        if s[0] == 1 and s[1] == 0 or s[1] == 3: #not possible when obstacle or end of grid
                            randomAct2 = [s[0], s[1]]
                        else:
                            randomAct2 = [s[0], s[1]+1]      

                    if a == 'S':
                        nextState = [s[0]+1, s[1]]

                         #if randomMove to the right of south (west)
                        if s[0] == 1 and s[1] == 2 or s[1] == 0: #not possible when obstacle or end of grid
                            randomAct1 = [s[0], s[1]]
                        else:
                            randomAct1 = [s[0], s[1]-1]

                        #if randomMove to the left of south (east)
                        if s[0] == 1 and s[1] == 0 or s[1] == 3: #not possible when obstacle or end of grid
                            randomAct2 = [s[0], s[1]]
                        else:
                            randomAct2 = [s[0], s[1]+1]    

                    if a == 'W':
                        nextState = [s[0], s[1]-1]

                        #if randomMove to the right of west (north)
                        if s[0] == 2 and s[1] == 1 or s[0] == 0: #not possible when obstacle or end of grid
                            randomAct1 = [s[0], s[1]]
                        else:
                            randomAct1 = [s[0]-1, s[1]]

                        #if randomMove to the left of west (south)
                        if s[0] == 0 and s[1] == 1 or s[0] == 2: #not possible when obstacle or end of grid
                            randomAct2 = [s[0], s[1]]
                        else:
                            randomAct2 = [s[0]+1, s[1]]

                    if a == 'E':
                        nextState = [s[0], s[1]+1]

                        #if randomMove to the left of east (north)
                        if s[0] == 2 and s[1] == 1 or s[0] == 0: #not possible when obstacle or end of grid
                            randomAct1 = [s[0], s[1]]
                        else:
                            randomAct1 = [s[0]-1, s[1]]

                    #if randomMove to the right of east (south)
                        if s[0] == 0 and s[1] == 1 or s[0] == 2: #not possible when obstacle or end of grid
                            randomAct2 = [s[0], s[1]]
                        else:
                            randomAct2 = [s[0]+1, s[1]]
                
                    #convert our coordinates
                    nextState = tuple(nextState)
                    randomAct1 = tuple(randomAct1)
                    randomAct2 = tuple(randomAct2)

                    #basing the reward on our probability distribution, 0.8 times the state we want to reach and the random moves
                    v = rewards[s] + gamma * ((0.8)* V[nextState] + (0.1 * V[randomAct1]) + (0.1 * V[randomAct2]))
                    #print("v", v)
                    #print("new V", new_v)
                    #print(" ")
                    #If the action we just took gives us the best value
                    if v > new_v: 
                        new_v = v #our new value of v will be the value we just computed
                        PI[s] = a #we change the policy to take the better action on that state

                #we update the value function, and check if we should stop the loop                             
                V[s] = new_v
                #print(variation) 
                variation = max(variation, np.abs(old_v - new_v))
                
        if variation < convergenceCondition:
            convergence = False

#valueIterationAlgorithm()
print("V")
print(V)
print("policy")
print(PI)

def Qlearning(alpha = 1, convergenceCondition = 0.005, explorationRate = 0.1, gamma = 1):
    convergence = True
    while convergence == True:
        variation = 0
        for s in states:
            old_Q = Q[s]   #old value of V(s) we need to check if we want to stop
            new_Q = 0 
            explorationExplotation = random.uniform(0, 1)
            if explorationExplotation > explorationRate: #we take the best action possible for the state
                for a in generalActions[s]:
                    if max(Q[s]) == 'N':
                        action = 'N'
                        nextState = [s[0]-1, s[1]]
                    if max(Q[s]) == 'S':
                        action = 'S'
                        nextState = [s[0]+1, s[1]]
                    if max(Q[s]) == 'W':
                        action = 'W'
                        nextState = [s[0], s[1]-1]
                    if max(Q[s]) == 'E':
                        action = 'E'
                        nextState = [s[0], s[1]+1]
            else: 
                for a in generalActions[s]:
                    randomAction = np.random.choice([i for i in generalActions[s]])
                    if randomAction == 'N':
                        action = 'N'
                        nextState = [s[0]-1, s[1]]
                    if randomAction == 'S':
                        action = 'S'
                        nextState = [s[0]+1, s[1]]
                    if randomAction == 'W':
                        action = 'W'
                        nextState = [s[0], s[1]-1]
                    if randomAction == 'E':
                        action = 'E'
                        nextState = [s[0], s[1]+1]

            nextState = tuple(nextState)
            #get following state and reward

            q = (1 - alpha) * Q[s][action] + alpha * (rewards[s] + gamma * Q[nextState][action]) 
            if q > new_Q:
                new_Q = q
                #Q[s][] = a
            Q[s][0] = new_Q

        variation = max(variation, np.abs(old_Q - new_Q))
        if variation < convergenceCondition:
            convergence = False

    return Q

#a = Q[states[2]]
#print(a)
#print(a[1])
#print(Q[states[2]]['S'])
#print(generalActions[states[3]])

#to access the value of a state we do Q[states[index]][0]
#to access the policy of a state we do Q[states[index]][1]
#print(max(tuple))
#print(Q[states[2]])
#print(max(Q[states[2]])) #to get the best action
print(Q)