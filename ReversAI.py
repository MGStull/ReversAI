
#Goals
#   1. MLP combined with Monte Carlos Tree Search with adjustable search depth
#   2. Train on Online Data sets and convert them to useful format
#   3. Download the Trained Model and implement it on the website

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random as rand
import DataProccess


class DenseLayer:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) *np.sqrt(2/input_size)
        self.B = np.zeros((1, output_size))
        self.dW = None
        self.dB = None

    def forward(self, x):
        self.x = np.copy(x)
        self.z = x@self.W + self.B
        self.a = self._apply_activation(self.z)
        return self.a

    #From Geeks for Geeks
    def _apply_activation(self,x,alpha=.01):
        return np.maximum(alpha * x, x)

    #From Geeks for Geeks
    def activationDerivative(self,x,alpha=.01):
        dx = np.ones_like(x)
        dx[x < 0] =alpha
        return dx

    def update(self,stepSize):
        self.W -= self.dW*stepSize
        self.B -= self.dB*stepSize

        

        
class Neural_Network:
    def __init__(self, layers_config,stepSize=.01):
        self.layers = []
        self.stepSize = stepSize
        for config in layers_config:
            self.layers.append(DenseLayer(*config))
    
    def guess(self, x):
        a = self.forward(x)
        return np.argmax(a)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backprop(self,batch):
        batch.predicted = self.forward(batch.input)
        self.weightGradient(batch,self.layers)
        for layer in self.layers:
            layer.update(self.stepSize)
        return self.error(batch)

    def error(self,batch):
        return np.linalg.norm(batch.predicted-batch.expected,ord=2)/batch.size
    def errorGradient(self,batch):
        return 2*(batch.predicted-batch.expected)/batch.size

    def weightGradient(self,batch,weights):
        delta = self.errorGradient(batch)
        for layer in reversed(self.layers):
            dz=delta*layer.activationDerivative(layer.z)
            dW = layer.x.T @ dz
            dB = np.sum(dz,axis=0,keepdims=True)
            delta = dz@layer.W.T
            layer.dW =dW
            layer.dB =dB


##CHANGE BATCH AND EPOCH DATA PROCCESSING

class batch:
        def __init__(self,indices,x_data,y_data,num_classes=64):
            self.size=len(indices)
            self.num_classes = num_classes
            self.input,self.expected = self.Grab(indices,x_data,y_data)
            self.indices = indices
        
        def Grab(self,indices,x_data,y_data):
            #changed to by 8 by 8
            X_batch = np.zeros((self.size,65))
            Y_batch = np.zeros(self.size,dtype=int)

            for i in range(0,len(indices)):    
                X_batch[i] = x_data[indices[i]].reshape(65)  
                Y_batch[i] = y_data[indices[i]]  

            Y_batch_one_hot = np.zeros((self.size, self.num_classes))
            Y_batch_one_hot[np.arange(self.size), Y_batch] = 1
            return [X_batch, Y_batch_one_hot]

class Epoch:
    def __init__(self,k,x_data,y_data,num_classes):
        self.indicesList = [[] for _ in range(k)]
        self.batches = []

        randHold = list(range(len(x_data)))
        rand.shuffle(randHold)
        chunk_size = len(x_data) // k
        
        for i in range(k - 1):
            self.indicesList[i] = randHold[i*chunk_size:(i+1)*chunk_size]
        self.indicesList[k - 1] = randHold[(k - 1)*chunk_size:]

        for i in self.indicesList:
            self.batches.append(batch(i,x_data,y_data,num_classes))

    def randTrain(self,NN):
        for i in self.batches:
            NN.backprop(i)
        #print("trained like magnus")

    def kmc(self,NN):
        k = len(self.batches)
        for i in range(len(self.batches)-1):
            NN.backprop(self.batches[i])
        return self.test(NN,self.batches[len(self.batches)-1])
    
    def test(self,NN,batch):
        sum=0
        for i,yTrue in enumerate(batch.expected):
            if NN.guess(batch.input[i]) == np.argmax(yTrue):
                sum+=1
        return sum/len(batch.input)

    
        



