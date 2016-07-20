import numpy as np

import glob
import os

def get_data(num, sampleIndex) :

    labels = np.loadtxt('Labels' + str(sampleIndex) + str(num) + '.txt')
    skeletons = np.loadtxt('Skeleton' + str(sampleIndex) + str(num) + '.txt')
    frames = np.zeros((len(skeletons) / 20, 20, 9))

    for i in range(len(skeletons) / 20):
        for j in range(20):
            frames[i] = skeletons[i*20 + j]

    return labels, frames

#lb, fr = get_data(1,151)

#path = r'C:\Users\Boris\Documents\Astronomija i Petnica uopšte\PSIML\Projekat\Fixed Input'
os.chdir("C:\\Users\\Boris\\Documents\\PSIML\\Projekat\\Fixed Input")


labels = np.empty([0,1], dtype=np.str)
for file in glob.glob("Labels*.txt"):
    labels = np.append(labels, file)
    
skeletons = np.epty([0,1], dtype = np.str)
for file in glob.glob("Skeletons*.txt"):
    skeletons = np.append(skeletons, file)
    


#TODO: Obavezno permutirati redosled Samplova, i to na isti način i za skeletone i za labele
#TODO: Najbolje prvo generisati permuzaciju niza dužine ukupnog broja Samplova, pa ga 
#      koristiti kao range za prolaženje kroz neki for ciklus    
#EDIT: Ovo je zapravo ova linija ispod!

pemutationArray = np.random.permutation(np.asarray(range(1,len(skeletons))))
 


