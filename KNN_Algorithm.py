
# coding: utf-8

# In[48]:

import csv
with open(r'C:\Users\shahina\Documents\iris.data.') as csvfile:
    lines=csv.reader(csvfile)
    for row in lines:
        print(','.join(row))


# In[50]:

import csv
import random
def loadDataset(filename,split,trainingSet=[],testSet=[]):
    with open(filename,'r') as csvfile:
        lines=csv.reader(csvfile)
        dataset=list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y]=float(dataset[x][y])
            if random.random()<split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


# In[52]:

trainingSet=[]
testSet=[]
loadDataset(r'C:\Users\shahina\Documents\iris.data.' ,0.66,trainingSet,testSet)
print('Train:'+repr(len(trainingSet)))
print('Test:'+repr(len(testSet)))


# In[53]:

import math
def euclideandistance(instance1,instance2,length):
    distance=0
    for x in range(length):
        distance+=pow((instance1[x]-instance2[x]),2)
    return math.sqrt(distance)


# In[54]:

data1=[2,2,2,'a']
data2=[4,4,4,'b']
distance=euclideandistance(data1,data2,3)
print('Distance:',repr(distance))


# In[59]:

import operator
def getneighbors(trainingSet,testInstance,k):
    distance=[]
    length=len(testInstance)-1
    for x in range(len(trainingSet)):
        dist=euclideandistance(testInstance,trainingSet[x],length)
        distance.append((trainingSet[x],dist))
    distance.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in range(k):
        neighbors.append(distance[x][0])
    return (neighbors)


# In[71]:

trainingSet=[[2,2,2,'a'],[4,4,4,'b']]
testInstance=[5,5,5]
k=1
neighbors=getneighbors(trainingSet,testInstance,1)
print(neighbors)


# In[80]:


import operator
def getResponse(neighbors):
    classVotes={}
    for x in range(len(neighbors)):
        response=neighbors[x][-1]
        if response in classVotes:
            classVotes[response]+=1
        else:
            classVotes[response]=1
    sortedvotes=sorted(classVotes.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedvotes[0][0]



neighbors=[[1,1,1,'a'],[2,2,2,'a'],[3,3,3,'b']]
response=getResponse(neighbors)
print(response)


# In[65]:

def getAccuracy(testSet,predictions):
    correct=0
    for x in range(len(testSet)):
        if testSet[x][-1] is predictions[x]:
            correct +=1
    return(correct/float(len(testSet)))*100.0



testSet=[[1,1,1,'a'],[2,2,2,'a'],[3,3,3,'b']]
predictions=['a','a','a']
accuracy=getAccuracy(testSet,predictions)
print(accuracy)


# In[ ]:

def main():
    trainingSet=[]
    testSet=[]
    split=0.67
    loadDataset('iris.data',split,trainingSet,testSet)
    print('trainset'+repr(len(trainingSet)))
    print('testset'+repr(len(testSet)))
    predictions=[]
    k=3
    for x in range(len(testSet)):
        neighbors=getneighbors(trainingSet,testSet[x],k)
        result=get
    

