import numpy as np

def split_data(dataset, train_size, test_size):
    X_train = list(map(lambda x: x[0], dataset[:train_size]))
    Y_train = list(map(lambda x: x[1], dataset[:train_size]))
    X_test = list(map(lambda x: x[0], dataset[train_size:train_size+test_size]))
    Y_test = list(map(lambda x: x[1], dataset[train_size:train_size+test_size]))
    
    X_train = np.concatenate(X_train, axis=1)
    X_test = np.concatenate(X_test, axis=1)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    
    return X_train, Y_train, X_test, Y_test

