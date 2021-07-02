#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
from io import StringIO
import matplotlib.pyplot as plt




a = open('trn.txt').read()




a = pd.read_csv(StringIO(a), header=None,delimiter = " ", sep ="", names= ["x",'y',"target"])




b = open('tst.txt').read()
b = pd.read_csv(StringIO(b), header=None,delimiter = " ", sep ="", names= ["x",'y'])




c=b.copy()




one = a[a["target"]==0]
zero = a[a["target"]==1]
data = a.drop("target", axis=1)




a["target"].value_counts()




plt.plot(one["x"], one["y"])
plt.plot(zero["x"], zero["y"])
plt.show()



def sigmoid(x):
    return 1 / (1 + np.exp(-x))




def der_sigmoid(a):
    return sigmoid(a)*(1- sigmoid(a))




def loss_function(final_output, true_values):
    squared_errors = (final_output - true_values ) ** 2
    return np.sum(squared_errors)




output_data =np.array(a["target"])
output_data= output_data.reshape((630,1))




input_data = np.array(a.drop(["target"],axis=1))




rate =0.1




def forward(X,w1,w2,w3,w4):
    first_layer = np.dot(X,w1)
    first_activation = sigmoid(first_layer)
    second_layer = np.dot(first_activation,w2)
    second_activation = sigmoid(second_layer)
    third_layer = np.dot(second_activation,w3)
    third_activation = sigmoid(third_layer)
    fourth_layer = np.dot(third_activation,w4)
    fourth_activation = sigmoid(fourth_layer)
    label = (fourth_activation>=0.5).astype(int)
    return label




losses_ar= []





def train(train,w1,w2,w3,w4):
        if len(train)==0:
            return False
        else:  
            X = train[:,[0,1]]
            Y = train[:,2].reshape(len(train), 1)

            first_layer = np.dot(X,w1)
            first_activation = sigmoid(first_layer)
            second_layer = np.dot(first_activation,w2)
            second_activation = sigmoid(second_layer)
            third_layer = np.dot(second_activation,w3)
            third_activation = sigmoid(third_layer)
            fourth_layer = np.dot(third_activation,w4)
            fourth_activation = sigmoid(fourth_layer)
            loss= loss_function(fourth_activation,Y)
            #print(loss)
            #losses_ar.append(loss)
            error = (fourth_activation-Y)
    
            d_4 = error * der_sigmoid(fourth_layer)
            d_act_3 = np.dot(d_4, w4.T)
            d_wx4= np.dot(third_activation.T,d_4)
            
            d_3 =  d_act_3*der_sigmoid(third_layer)
            d_act_2= np.dot( d_3,w3.T)
            d_wx3 =  np.dot(second_activation.T,d_3)

            d_2 =  d_act_2*der_sigmoid(second_layer)
            d_act_1= np.dot( d_2,w2.T)
            d_wx2 =  np.dot(first_activation.T,d_2)

            d_1 = d_act_1 * der_sigmoid(first_layer)
            d_wx1 = np.dot(X.T,d_1)
            
            w1-=d_wx1*rate/X.shape[0]
            w2-=d_wx2* rate/X.shape[0]
            w3-=d_wx3* rate/X.shape[0]
            w4-=d_wx4* rate/X.shape[0]
            
            return loss




def SGD(data, epochs, mini_batch_size, test):
    weight1 = np.random.uniform(size=(2,20))
    weight2= np.random.uniform(size=(20,10))
    weight3 = np.random.uniform(size=(10,6))
    weight4 = np.random.uniform(size=(6,1))
    
    n= len(data)
    for i in range(epochs):
        np.random.shuffle(data)
        mini_batches = [data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
        a=[]
        for m in mini_batches:
            l=train(m,weight1,weight2,weight3,weight4)
        print("Epoch {0} {1}".format(i, l)) 
        losses_ar.append(l)
    return forward(test, weight1,weight2,weight3,weight4)




losses_ar=[]
predict = SGD(np.array(a), 2000, 1, np.array(c))




plt.plot(np.arange(1, np.array(losses_ar).shape[0] + 1), np.array(losses_ar))
plt.show()



b["predict"]=predict
b["predict"].value_counts()

p_one = b[b["predict"]==0]
p_zero = b[b["predict"]==1]
plt.scatter(p_one["x"], p_one["y"])
plt.scatter(p_zero["x"], p_zero["y"])

plt.scatter(one["x"], one["y"])
plt.scatter(zero["x"], zero["y"])

























