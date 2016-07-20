import numpy as np

import glob
import os

def get_data(labelName, skeletonName) :
    os.chdir("C:\\Users\\Boris\\Documents\\PSIML\\Projekat\\Debug Inputs")
    labels = np.loadtxt(labelName)
    skeletons = np.loadtxt(skeletonName)
    #print(size)
    #raw_input()
    frames = np.zeros((len(skeletons) / 20, 20, 7))

    for i in range(len(skeletons) / 20):
        frames[i] = skeletons[i*20:i*20 + 20]

    return labels, frames

#lb, fr = get_data(1,151)
os.chdir("C:\\Users\\Boris\\Documents\\PSIML\\Projekat\\Debug Inputs")


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
        position = position + 1
        
