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
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

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


def feature_extract(directory):
    os.chdir(directory)    
    labelFiles = np.empty([0,1], dtype=np.str)
    for file in glob.glob("Labels*.txt"):
        labelFiles = np.append(labelFiles, file)
        
    skeletonFiles = np.empty([0,1], dtype = np.str)
    for file in glob.glob("Skeletons*.txt"):
        skeletonFiles = np.append(skeletonFiles, file)
        
    
    #Vectorization
    permutationArray = np.random.permutation(np.asarray(range(len(skeletonFiles))))
    slidingWidth = 10
    
    
    
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
X,Y = feature_extract(dir)

validDir = "C:\\Users\\Boris\\Documents\\PSIML\\Projekat\\Validation Inputs"

X_valid = X[len(X)-20*140:len(X)-1]
Y_valid = Y[len(Y)-20*140:len(Y)-1]

X = X[0:len(X)-20*140]
Y = Y[0:len(Y)-20*140]

print(len(X))

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(21, 20*7)))
model.add(Dense(3))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

for iteration in range(1, 10):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, Y, batch_size=32, nb_epoch=1)

    score, acc = model.evaluate(X, Y, batch_size=32)
    print('Test score:', score)
    print('Test accuracy:', acc)

    