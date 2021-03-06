# -*- coding: utf-8 -*-
"""GA plusz GD kombo CIFAR

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1b6ALI8axNGXwLX_71ZPZ84TtMSLHNgCe

# **Adatok betöltése és könyvtárak importálása**
"""

# Könyvtárak importálása
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras import Sequential, layers, models, datasets
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
import datetime
import seaborn as sn
import pandas as pd
import random, time

input_height=32
input_width=32
batch_size=32
IMG_SIZE = (input_height,input_width)

"""# **Tanító és validációs adatgyűjtemények létrehozása**"""

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

"""# **Modell létrehozása**

Az összevonó réteg láthatóan csökkenti az adataugmentáció hatását
"""

def modelbuilder():
    model = keras.Sequential([
      layers.Input(shape=(32, 32, 3)),
      layers.Conv2D(8, (3, 3), activation='sigmoid'),
      layers.MaxPool2D(),
      layers.Conv2D(8, (3, 3), activation='sigmoid'),
      layers.MaxPool2D(),
      layers.Conv2D(8, (3, 3), activation='sigmoid'),
      layers.MaxPool2D(),
      layers.Flatten(),
      layers.Dense(len(class_names),activation="softmax")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])
    return model

main_model=modelbuilder()

main_model.summary()

historyGD=main_model.fit(train_images,train_labels,validation_data=(test_images,test_labels), epochs=20,verbose=1)

def model_crossover(model1, model2):#öröklődés 2 szülőtől
    model=modelbuilder()
    #mutation_rate=0.9
    #mutation_power=4
    egyikmasik=0.5
    #for each layer
    w1=model1.get_weights()
    w2=model2.get_weights()
    #print(type(w1))    
    #print((w1)[0].shape)
    wf=[]
    for x in range(len(w1)):
        if len(w1[x].shape)==4:
            #print("4")
            xx=np.random.rand(w1[x].shape[0],w1[x].shape[1],w1[x].shape[2],w1[x].shape[3])    
            #mutation_choise=np.random.rand(w1[x].shape[0],w1[x].shape[1],w1[x].shape[2],w1[x].shape[3])    
            #mutation_val=(np.random.rand(w1[x].shape[0],w1[x].shape[1],w1[x].shape[2],w1[x].shape[3])-0.5)*mutation_power          
            z = np.where(xx>egyikmasik, w1[x], w2[x])
            #z = np.where(mutation_choise>mutation_rate,mutation_val*z,z)
            wf.append(z)
        if len(w1[x].shape)==3:
            #print("3")
            xx=np.random.rand(w1[x].shape[0],w1[x].shape[1],w1[x].shape[2])    
            #mutation_choise=np.random.rand(w1[x].shape[0],w1[x].shape[1],w1[x].shape[2])    
            #mutation_val=(np.random.rand(w1[x].shape[0],w1[x].shape[1],w1[x].shape[2])-0.5)*mutation_power           
            z = np.where(xx>egyikmasik, w1[x], w2[x])
            #z = np.where(mutation_choise>mutation_rate,mutation_val*z,z)
            wf.append(z)
        if len(w1[x].shape)==2:
            #print("2")
            xx=np.random.rand(w1[x].shape[0],w1[x].shape[1])    
            #mutation_choise=np.random.rand(w1[x].shape[0],w1[x].shape[1])    
            #mutation_val=(np.random.rand(w1[x].shape[0],w1[x].shape[1])-0.5)*mutation_power          
            z = np.where(xx>egyikmasik, w1[x], w2[x])
            #z = np.where(mutation_choise>mutation_rate,mutation_val*z,z)
            wf.append(z)
        if len(w1[x].shape)==1:
            #print("1")
            xx=np.random.rand(w1[x].shape[0])    
            #mutation_choise=np.random.rand(w1[x].shape[0])    
            #mutation_val=(np.random.rand(w1[x].shape[0])-0.5)*mutation_power           
            z = np.where(xx>egyikmasik, w1[x], w2[x])
            #z = np.where(mutation_choise>mutation_rate,mutation_val*z,z)      
            #z=np.asscalar(z)
            wf.append(z)
            #print(wf)
                                        
    model.set_weights(wf)
 
    return model

def mutation(model1,mutation_rate,mutation_power):
    model=modelbuilder()
    #for each layer
    w1=model1.get_weights()
   #w2=model2.get_weights()
    #print(type(w1))    
    #print((w1)[0].shape)
    wf=[]
    for x in range(len(w1)):
        if len(w1[x].shape)==4:
            #print("4")
            #xx=np.random.rand(w1[x].shape[0],w1[x].shape[1],w1[x].shape[2],w1[x].shape[3])    
            mutation_choise=np.random.rand(w1[x].shape[0],w1[x].shape[1],w1[x].shape[2],w1[x].shape[3])    
            mutation_val=(np.random.rand(w1[x].shape[0],w1[x].shape[1],w1[x].shape[2],w1[x].shape[3])-0.5)*mutation_power          
            z = np.where(True, w1[x], w1[x])
            z = np.where(mutation_choise>mutation_rate,mutation_val+z,z)
            wf.append(z)
        if len(w1[x].shape)==3:
            #print("3")
           # xx=np.random.rand(w1[x].shape[0],w1[x].shape[1],w1[x].shape[2])    
            mutation_choise=np.random.rand(w1[x].shape[0],w1[x].shape[1],w1[x].shape[2])    
            mutation_val=(np.random.rand(w1[x].shape[0],w1[x].shape[1],w1[x].shape[2])-0.5)*mutation_power           
            z = np.where(True, w1[x], w1[x])
            z = np.where(mutation_choise>mutation_rate,mutation_val+z,z)
            wf.append(z)
        if len(w1[x].shape)==2:
            #print("2")
            #xx=np.random.rand(w1[x].shape[0],w1[x].shape[1])    
            mutation_choise=np.random.rand(w1[x].shape[0],w1[x].shape[1])    
            mutation_val=(np.random.rand(w1[x].shape[0],w1[x].shape[1])-0.5)*mutation_power          
            z = np.where(True, w1[x], w1[x])
            z = np.where(mutation_choise>mutation_rate,mutation_val+z,z)
            wf.append(z)
        if len(w1[x].shape)==1:
            #print("1")
            #xx=np.random.rand(w1[x].shape[0])    
            mutation_choise=np.random.rand(w1[x].shape[0])    
            mutation_val=(np.random.rand(w1[x].shape[0])-0.5)*mutation_power           
            z = np.where(True, w1[x], w1[x])
            z = np.where(mutation_choise>mutation_rate,mutation_val+z,z)      
            #z=np.asscalar(z)
            wf.append(z)
            #print(wf)
                                        
    model.set_weights(wf)
 
    return model

def runtournament():#random kiválaszt párat és megnézi h azok közül melyik a legjobb 2
    list_idx_on_tournament=[]
    for x in range (tournament_sel):
        list_idx_on_tournament.append(int(random.uniform(0, total_models-1)))
        
    best1=99999
    best2=99999
    best1_idx=-1
    best2_idx=-1
    for  x in range (tournament_sel):
        if fitness[list_idx_on_tournament[x]]<best1:
            best1=fitness[x]
            best1_idx=x
            
    for  x in range (tournament_sel):
        if fitness[list_idx_on_tournament[x]]>best2 and x!=best1_idx:
            best2=fitness[x]
            best2_idx=x        
    return current_pool[list_idx_on_tournament[best1_idx]],current_pool[list_idx_on_tournament[best2_idx]]

def parallel_scoring(x,images,labels):#fitnesst számol
    score=0
    score =current_pool[x].evaluate(images,labels, return_dict=True, verbose=0)
    fitness=score['loss']
    return fitness

def parallel_muttion(i,mutation_rate,mutation_power):
    model1,model2=runtournament()#véletlenszerűen pár a meglévők közül a 2 legjobb
    model=model_crossover(model1,model2)# azon 2-nek a kombója valahogyan
    model=mutation(model,mutation_rate,mutation_power)
    return model

current_pool = [] #actual networks saved
fitness = [] #save value for each network
total_models = 8 
generations=20 
bestfitness_index=0
tournament_sel=3 #usefull for tournament selection method
        
    
# Initialize all models with random weigth
for i in range(total_models):
    # model
    model=modelbuilder()
    weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)
    #model.summary()
    current_pool.append(model)
    fitness.append(-100)

val_scores=[]
val_acc=[]
train_scores=[]
train_acc=[]

for i in range(generations):
        print("=========================Generation:",i)
        results=[]
        new_pool=[]
        #print("Train current pool and scoring:")
        for x in current_pool:
          #history=current_pool[x].fit(x_train,y_train, epochs=1)
          history=x.fit(train_images,train_labels, epochs=1)
          results.append(history.history["loss"][-1])

        #print("Train scoring")
        #for x in range(total_models):
         #   results.append(parallel_scoring(x,x_train, y_train))
        train_score=0
        val_score=0
        fitness=results
        print(fitness)
        bestfitness_index = np.argmin(np.array(fitness)) 
        print ('Best fitness score and best fitness index: ',fitness[bestfitness_index],bestfitness_index)
        best_model=current_pool[bestfitness_index]#we save this model and save this as last
        #Performance on the train data
        train_score=best_model.evaluate(train_images, train_labels, return_dict=True, verbose=0)
        print("train_loss: ",train_score['loss'])
        print("train_acc: ", train_score['accuracy'])
        train_scores.append(train_score['loss'])
        train_acc.append(train_score['accuracy'])
        #Performance on the val data
        val_score=best_model.evaluate(test_images, test_labels, return_dict=True, verbose=0)
        print("val_loss: ",val_score['loss'])
        print("val_acc: ", val_score['accuracy'])
        val_scores.append(val_score['loss'])
        val_acc.append(val_score['accuracy'])

              
        for x in range(total_models-1):
            model=parallel_muttion(i,0.99,0.4)
            new_pool.append(model) 
                                      
        current_pool=new_pool
        current_pool.append(best_model)
        
            

print ('The final best fitness score and best fitness index: ',fitness[bestfitness_index],bestfitness_index)

"""# **Kiértékelés**"""

plt.figure(figsize=(5,5))
plt.plot(range(len(historyGD.history['loss'])), historyGD.history['loss'], label='Gradient Descent')
plt.plot(range(len(train_scores)), train_scores, label='GD+GA')

plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.figure(figsize=(5,5))
plt.plot(range(len(historyGD.history['accuracy'])), historyGD.history['accuracy'], label='Gradient Descent')
plt.plot(range(len(train_acc)), train_acc, label='GD+GA')

plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')