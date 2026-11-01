import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Dense, LSTM
import warnings
warnings.filterwarnings('ignore')

class sequential_lstm:
    def __init__(self, x, y):
        self.x = x
        self.y= y

    def one_to_one_lstm(self):
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', input_shape=(1,1)))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')
        print(self.model.summary())
        self.model.fit(self.x, self.y, batch_size=5, epochs=200, validation_split=0.2, verbose=0)
        return self.model

    def one_to_one_stacked_lstm(self):
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(1,1)))
        self.model.add(LSTM(50, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')
        print(self.model.summary())
        self.model.fit(self.x, self.y, batch_size=5, epochs=2000, validation_split=0.2, verbose=0)
        return self.model
    
    
if __name__ == '__main__':

    X = [i for i in range(1, 21)]
    y = [10*i for i in X]
    X = np.array(X).reshape(20, 1, 1)
    y = np.array(y)

    x_test = np.array([30])
    print(x_test)
    x_test = x_test.reshape((1,1,1))
    print(x_test)

    lstm_model = sequential_lstm(X, y)

    model = lstm_model.one_to_one_lstm()
    y_test = model.predict(x_test, verbose=0)
    print(y_test)

    model = lstm_model.one_to_one_stacked_lstm()
    y_test = model.predict(x_test, verbose=0)
    print(y_test)

    X1 = [2*i for i in range(1, 21)]
    X2 = [3*i for i in range(1, 21)]
    y = [x1*x2 for (x1, x2) in zip(X1,X2)]
    


