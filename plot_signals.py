import matplotlib.pyplot as plt
from sklearn import preprocessing


def plot_data(signal_mat, fig_name, range_size):
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(signal_mat.T)
    signal_mat_scaled = scaler.transform(signal_mat.T)
    signal_mat_scaled = signal_mat_scaled.T
    #signal_mat_scaled = signal_mat
    T = range(range_size)
    f, axarr = plt.subplots(signal_mat_scaled.shape[0], sharex=True)
    #print((signal_mat[0, 0:(range_size-1)]).shape)
    for i in range(signal_mat_scaled.shape[0]):
        axarr[i].plot(T, signal_mat_scaled[i, 0:range_size])
        axarr[i].set_title('Signal '+str(i))
        #axarr[i].axis([0,range_size,0,1])

    plt.savefig('figures/'+fig_name+str(range_size)+'.png')
    plt.show()
