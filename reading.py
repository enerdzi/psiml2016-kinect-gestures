import numpy as np

def get_data(num) :

    labels = np.loadtxt('Labels' + num + '.txt')
    skeletons = np.loadtxt('Skeleton' + num + '.txt')
    frames = np.zeros((len(skeletons) / 20, 20, 9))

    for i in range(len(skeletons) / 20):
        for j in range(20):
            frames[i] = skeletons[i*20 + j]

    return labels, frames

lb, fr = get_data('00151')