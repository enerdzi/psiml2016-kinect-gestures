'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import Callback
from keras.utils.data_utils import get_file
from keras.callbacks import History
import numpy as np
import random
import sys

import time
import matplotlib.pyplot as plt
import glob
import os


def get_data(labelName, skeletonName) :
    #os.chdir("C:\\Users\\Boris\\Documents\\PSIML\\Projekat\\Latest Inputs")
    labels = np.loadtxt(labelName)
    skeletons = np.loadtxt(skeletonName)
    #print(size)
    #raw_input()
    frames = np.zeros((len(skeletons) / 20, 20, 7))

    for i in range(len(skeletons) / 20):
        frames[i] = skeletons[i*20:i*20 + 20]

    return labels, frames

#lb, fr = get_data(1,151)


def feature_extract(directory, slidingWidth, stepSize):
    os.chdir(directory)    
    labelFiles = np.empty([0,1], dtype=np.str)
    for file in glob.glob("Labels*.txt"):
        labelFiles = np.append(labelFiles, file)
        
    skeletonFiles = np.empty([0,1], dtype = np.str)
    for file in glob.glob("Skeletons*.txt"):
        skeletonFiles = np.append(skeletonFiles, file)
        
    
    #Vectorization
    permutationArray = np.random.permutation(np.asarray(range(len(skeletonFiles))))
    
    count = 0
    for i in range(len(permutationArray)):
        labels, frames = get_data(labelFiles[permutationArray[i]], skeletonFiles[permutationArray[i]])
        count = count + len(frames) // stepSize
    
    X = np.zeros([count, 2*slidingWidth + 1, 20*7], dtype = float)
    Y = np.zeros([count, 3], dtype = float)
    position = 0
    print("Feature extraction")
    for i in range(len(permutationArray)):
        if(i % 50 == 0):
            print(round(100.0*float(i)/len(permutationArray)), "%")
        labels, frames = get_data(labelFiles[permutationArray[i]], skeletonFiles[permutationArray[i]])
        
        for j in range(len(frames) // stepSize):
            for k in range(2*slidingWidth + 1):
                frame_info = np.zeros([20*7])            
                if (not((j*stepSize - slidingWidth + k < 0) or (j*stepSize - slidingWidth + k > len(frames) - 1))):

                    for l in range(20):
                        frame_info[l*7:(l+1)*7] = frames[j*stepSize - slidingWidth + k,l]
                X[position, k] = frame_info
            if(labels[j*stepSize] == 0):
                Y[position, 0] = 1
            elif(labels[j*stepSize] == 1):
                Y[position, 1] = 1
            else:
                Y[position, 2] = 1
            position = position + 1
    return X,Y
    
#--------------------------------------------------------------------------
#Input directory:
dir = "C:\\Users\\Boris\\Documents\\PSIML\\Projekat\\Latest Inputs"
#Feature parameters
slidingWidthArray = [10, 20, 40]
numOfGestures = 3   #Including junk group
stepSizeArray = [5, 10, 20]
#Arch parameters:
LTSMneurons = 64
DenseNeurons = 32
#Training parameters:
epochs = 40
#Output directory:
savedir = "C:\\Users\\Boris\\Documents\\PSIML\\Projekat\\Prezentacija\\"

#TODO: Add arch header file!


for slidingWidth in slidingWidthArray:
    for stepSize in stepSizeArray:
        saveString = "_SW" + str(slidingWidth) + "_SS" + str(stepSize) + "_" + str(epochs) + "EP_"
        #Generating features from files in the input directory:
        X,Y = feature_extract(dir, slidingWidth, stepSize)
        validRation = 0.2
        X_valid = X[len(X)-int(round(0.2*len(X))) : len(X)-1]
        Y_valid = Y[len(Y)-int(round(0.2*len(Y))) : len(Y)-1]
        X = X[0 : int(round(0.8*len(X)))]
        Y = Y[0 : int(round(0.8*len(Y)))]
        
        # build the model: 1 stacked LSTM with 1 Dense layer
        print('Build model...')
        model = Sequential()
        #model.add(LSTM(128, activation='relu', input_shape=(21, 20*7)))
        model.add(LSTM(LTSMneurons, input_shape=(2*slidingWidth + 1, 20*7)))
        #model.add(Dropout(0.2))
        model.add(Dense(DenseNeurons))
        model.add(Dense(numOfGestures))
        model.add(Activation('softmax'))
        
        #optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        #for iteration in range(1):
        print()
        print('-' * 50)
        #print('Iteration', iteration)
        history = History()
        model.fit(X, Y, batch_size=32, nb_epoch=epochs,  show_accuracy=True, validation_data=(X_valid, Y_valid), callbacks=[history], verbose = 2)
        #time.sleep(0.1)
        loss_and_metrics = model.evaluate(X_valid, Y_valid, batch_size=32)
        
        #Saving model and weights
        model_architecture = model.to_json()
        model_architectureFile = savedir + "ARC" + saveString + ".json"
        with open(model_architectureFile, 'w') as output:
            output.write(model_architecture)
        
        weightsFile = savedir + "WEIGHTS" + saveString + ".h5"
        model.save_weights(weightsFile)
            
        #Prediction:
        predict = np.zeros([len(X_valid), numOfGestures], dtype = float)
        predictionInput = np.zeros([1,2*slidingWidth + 1, 20*7], dtype = float)
        for i in range(len(X_valid)):    
            predictionInput[0] = X_valid[i]    
            predict[i] = model.predict(predictionInput, verbose=0)[0]
        
        #Generating confusion Matrix:         
        #rows: predicted classes
        #clolumns: expected classes   
        confusionMatrix = np.zeros([numOfGestures, numOfGestures], dtype = float)
        trsh = 0.5
        finalPredictions = np.zeros([len(X_valid), numOfGestures], dtype = float)
        for i in range(len(predict)):
            for j in range(numOfGestures):
                if (predict[i,j] > trsh):
                    finalPredictions[i,j] = 1
            if (max(finalPredictions[i]) < 0.5):
                finalPredictions[i, 0] = 1
            cm_pred = np.nonzero(finalPredictions[i])[0][0]
            cm_lab = np.nonzero(Y_valid[i])[0][0]
            confusionMatrix[cm_pred, cm_lab] += 1
            
        confusionMatrix = confusionMatrix / len(predict)
        
        normalizedCM_predictionBased = np.zeros([numOfGestures, numOfGestures], dtype = float)
        normalizedCM_expectationBased = np.zeros([numOfGestures, numOfGestures], dtype = float)
        for i in range(3):
            normalizedCM_predictionBased[i,:] = confusionMatrix[i,:] / np.sum(confusionMatrix[i,:])
            normalizedCM_expectationBased[:,i] = confusionMatrix[:,i] / np.sum(confusionMatrix[:,i])
          
        matrixSave = savedir + "CM" + saveString
        np.savetxt(matrixSave + ".csv", confusionMatrix, delimiter = ",")
        loss_and_accSave = savedir + "LOSS_ACC" + saveString
        np.savetxt(loss_and_accSave + ".cvs", loss_and_metrics, delimiter = ",")
          
        #Visualizing results:  
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        pltSave = savedir + "ACC" + saveString
        plt.savefig(pltSave)
        plt.show()
        
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        pltSave = savedir + "LOSS" + saveString
        plt.savefig(pltSave)
        plt.show()
