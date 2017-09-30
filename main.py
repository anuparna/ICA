import processData as l
import mixData as m
import gradient_descent as gd
import plot_signals as p


def processTestData():
    # ------------- Work on test data obtained from icaTest.mat -----------------
    A, U = l.loadTestData()
    p.plot_data(U, 'original-test-signals', range_size=U.shape[1])

    X = m.mixTestData(A, U)
    # scaled_X = l.scale_data(X)
    p.plot_data(X, 'mixed-test-signals', range_size=X.shape[1])

    lossHistory, Y = gd.grad_descent(X, U, total_iterations=1000000, step_size=0.01)
    # print(lossHistory)
    p.plot_data(Y, 'recovered-test-signals', range_size=Y.shape[1])


def processAudioData():
    # ------------- Work on actual data ------------------------------
    original_signals = l.loadData()
    p.plot_data(original_signals, 'original-data-signals', range_size=1000)
    # Normalize so that it can be written back to file
    original_signals_file = l.scale_data(original_signals)
    l.createAudioFile(original_signals_file, 'original')

    # Mix them
    X = m.mixData(original_signals)
    p.plot_data(X, 'mixed-data-signals', range_size=1000)
    mixed_signals_file = l.scale_data(X)
    l.createAudioFile(mixed_signals_file, 'mixed')

    # Trying to recover original signals
    lossHistory, Y = gd.grad_descent(X, original_signals, total_iterations=200000, step_size=0.0003, isUniform=False)
    print(lossHistory)
    print(Y)
    p.plot_data(Y, 'recovered-data-signals', range_size=1000)
    recovered_signals_file = l.scale_data(Y)
    l.createAudioFile(recovered_signals_file, 'recovered')


if __name__ == '__main__':
    #processTestData()
    processAudioData()
