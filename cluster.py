import numpy as np
import matplotlib.pyplot as plt

size = 30 

#Calculate the distance between 2 points
def distEuclid(x,y):
    return np.sqrt(np.sum((x-y)**2))

#randomly generate the data set
def genDataset(n,dim):
    data = []
    while len(data)<n:
        p = np.around(np.random.rand(dim)*size,decimals=2)
        data.append(p) 
    return data

#Initializa the centroid
def initCentroid(data,k):
    num,dim = data.shape
    centpoint = np.zeros((k,dim))
    l = [x for x in range(num)]
    np.random.shuffle(l)
    for i in range(k):
        index = int(l[i])
        centpoint[i] = data[index]
    return centpoint

#K Mean Method
def KMeans(data,k):
    # Number of cells
    num = np.shape(data)[0]
    cluster = np.zeros((num,2))
    cluster[:,0]=-1
    
    # Keep track of the changed data
    change = True
    # Initializa the centroid
    cp = initCentroid(data,k)



    while change:
        change = False
        
        #Loop through all the data
        for i in range(num):
            minDist = 9999.9
            minIndex = -1
            
            # Calculate the shorest path
            for j in range(k):
                dis = distEuclid(cp[j],data[i])
                if dis < minDist:
                    minDist = dis
                    minIndex = j
            
            # Change to another cluster
            if cluster[i,0]!=minIndex:
                change = True
                cluster[i,:] = minIndex,minDist
        
        # Redraw the centriod
        for j in range(k):
            pointincluster = data[[x for x in range(num) if cluster[x,0]==j]]
            cp[j] = np.mean(pointincluster,axis=0)
    print("finish!")
    return cp,cluster

# Show the graph of the final result
def Show(data,k,cp,cluster):
    num,dim = data.shape
    color = ['r','g','b','c','y','m','k']

    for i in range(num):
        mark = int(cluster[i,0])
        plt.plot(data[i,0],data[i,1],color[mark]+'o')

    for i in range(k):
        plt.plot(cp[i,0],cp[i,1],color[i]+'x')


num,dim = data.shape
color = ['r','g','b','c','y','m','k']

num = 50 
k=4 
data = np.array(genDataset(num,2))
cp,cluster = KMeans(data,k)
Show(data,k,cp,cluster)
plt.show()
