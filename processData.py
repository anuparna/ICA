import scipy.io
import numpy as np
from scipy.io.wavfile import write


def loadData():
    data = scipy.io.loadmat('input/sounds.mat')['sounds']
    return data


def scale_data(data_matrix):
    data_matrix /= np.max(np.abs(data_matrix), axis=0)
    return data_matrix


def createAudioFile(data_matrix, directory_name):
    for i in range(data_matrix.shape[0]):
        write('audio/'+directory_name+'/signal'+str(i)+'.wav', 11025, data_matrix[i, :])


def loadTestData():
    data = scipy.io.loadmat('input/icaTest.mat')
    for name, variable_data in data.items():
        if not name.startswith('__'):
            if name == 'A':
                A = variable_data
            else:
                U = variable_data

    return A, U
