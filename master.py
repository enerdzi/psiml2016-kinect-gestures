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


def feature_extract(directory, slidingWidth):
    os.chdir(directory)    
    labelFiles = np.empty([0,1], dtype=np.str)
    for file in glob.glob("Labels*.txt"):
        labelFiles = np.append(labelFiles, file)
        
    skeletonFiles = np.empty([0,1], dtype = np.str)
    for file in glob.glob("Skeletons*.txt"):
        skeletonFiles = np.append(skeletonFiles, file)
        
    
    #Vectorization
    permutationArray = np.random.permutation(np.asarray(range(len(skeletonFiles))))
    
    #X1 = np.zeros([len(permutationArray), 1 + 2*slidingWidth, 20 * 7], dtype=float)
    
    #X = np.empty([1, 2*slidingWidth + 1, 20*7], dtype = float)
    
    count = 0
    for i in range(len(permutationArray)):
        labels, frames = get_data(labelFiles[permutationArray[i]], skeletonFiles[permutationArray[i]])
        count = count + len(frames)
    
    X = np.zeros([count, 2*slidingWidth + 1, 20*7], dtype = float)
    Y = np.zeros([count, 3], dtype = float)
    position = 0
    for i in range(len(permutationArray)):
        labels, frames = get_data(labelFiles[permutationArray[i]], skeletonFiles[permutationArray[i]])
        for j in range(len(frames)):
            for k in range(2*slidingWidth + 1):
                frame_info = np.zeros([20*7])            
                if (not((j - slidingWidth + k < 0) or (j - slidingWidth + k > len(frames) - 1))):
                #    frame_info = np.zeros([20*7])
                #else:
                    for l in range(20):
                        frame_info[l*7:(l+1)*7] = frames[j - slidingWidth + k,l]
                X[position, k] = frame_info
            if(labels[j] == 0):
                Y[position, 0] = 1
            elif(labels[j] == 1):
                Y[position, 1] = 1
            else:
                Y[position, 2] = 1
            position = position + 1
    return X,Y

dir = "C:\\Users\\Boris\\Documents\\PSIML\\Projekat\\Latest Inputs"
slidingWidth = 10
numOfGestures = 3
X,Y = feature_extract(dir,slidingWidth)

validDir = "C:\\Users\\Boris\\Documents\\PSIML\\Projekat\\Validation Inputs"

X_valid = X[len(X)-20*140:len(X)-1]
Y_valid = Y[len(Y)-20*140:len(Y)-1]

X = X[0:len(X)-20*140]
Y = Y[0:len(Y)-20*140]

print(len(X))

# build the model: 1 stacked LSTM
print('Build model...')
model = Sequential()
#model.add(LSTM(128, activation='relu', input_shape=(21, 20*7)))
model.add(LSTM(128, input_shape=(2*slidingWidth + 1, 20*7)))
model.add(Dense(numOfGestures))
model.add(Activation('softmax'))

#optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#for iteration in range(1):
print()
print('-' * 50)
#print('Iteration', iteration)
history = History()
model.fit(X, Y, batch_size=32, nb_epoch=1,  show_accuracy=True, validation_data=(X_valid, Y_valid), callbacks=[history])
loss_and_metrics = model.evaluate(X_valid, Y_valid, batch_size=32)

json_string = model.to_json()
    
#Prediction:
predict = np.zeros([len(X_valid), numOfGestures], dtype = float)
predictionInput = np.zeros([1,2*slidingWidth + 1, 20*7], dtype = float)
for i in range(len(X_valid)):    
    predictionInput[0] = X_valid[i]    
    predict[i] = model.predict(predictionInput, verbose=0)[0]

            
confusionMatrix = np.zeros([numOfGestures, numOfGestures], dtype = float)

trsh = 0.5
finalPredictions = np.zeros([len(X_valid), numOfGestures], dtype = float)
for i in range(len(predict)):
    for j in range(numOfGestures):
        if (predict[i,j] > trsh):
            finalPredictions[i,j] = 1
    if (max(finalPredictions[i] < 1)):
        finalPredictions[i, 0] = 1
    cm_pred = np.nonzero(predict[i])[0][0]
    cm_lab = np.nonzero(Y_valid[i])[0][0]
    confusionMatrix[cm_pred, cm_lab] += 1
    
#confusionMatrix/=len(predict)
    
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
