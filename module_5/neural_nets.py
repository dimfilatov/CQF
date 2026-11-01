import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'      # quieten TensorFlow's C++ logging

from random import shuffle
import warnings
warnings.filterwarnings('ignore')

# base libraries
import numpy as np
import pandas as pd
from pathlib import Path

# quantmod: bundled market data
from quantmod.datasets import fetch_historical_data

# scikit-learn modules
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# keras 3
import keras
from keras import layers

# plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

# reproducibility: seeds Python, NumPy and the backend in one call
keras.utils.set_random_seed(2026)

class NN:

    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_data(self):
        raw = fetch_historical_data(
                                    self.config['symbol'],
                                    start_date=self.config['start'],
                                    end_date=self.config['end']
                                    )

        data = raw.set_index(pd.to_datetime(raw['date']))[['close']]

        print(f'Shape: {data.shape} | Missing: {data.isna().sum().sum()} | {data.index[0].date()} to {data.index[-1].date()}')

        # Chronological split
        n = len(data)
        i_val = int(n * self.config['train_frac'])
        i_test = int(n * (self.config['train_frac'] + self.config['val_frac']))

        self.train_data = data.iloc[:i_val]
        self.val_data = data.iloc[i_val:i_test]
        self.test_data = data.iloc[i_test:]

        for name, df in [('Train', self.train_data), ('Validation', self.val_data), ('Test', self.test_data)]:
            print(f'{name:<11}{len(df):>6} days  {df.index[0].date()} to {df.index[-1].date()}')

        # Price history, the three blocks, and the highest level the model sees in training
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(data.index, data['close'], lw=1, color='black', label='S&P 500')
        ax.axvspan(self.val_data.index[0], self.val_data.index[-1], color='tab:orange', alpha=0.15, label='Validation')
        ax.axvspan(self.test_data.index[0], self.test_data.index[-1], color='tab:green', alpha=0.15, label='Test')
        ax.axhline(self.train_data['close'].max(), color='tab:red', ls='--', lw=1, label='Training maximum')
        ax.set(title='S&P 500 daily close', ylabel='Index level')
        ax.legend(loc='upper left')

        # Scale with MinMax fitted on training data only
        scaledtrain = self.scaler.fit_transform(self.train_data)
        scaledval = self.scaler.transform(self.val_data)
        scaledtest = self.scaler.transform(self.test_data)

        print(f'Scaled maximum | train: {scaledtrain.max():.2f} | validation: {scaledval.max():.2f} | test: {scaledtest.max():.2f}')

        def make_windows(series, seqlen, batch_size, shuffle=False):
            """Slide a window of `seqlen` steps over `series`; the target is the value right after each window."""
            return keras.utils.timeseries_dataset_from_array(
                data=series,
                targets=series[seqlen:],      # targets[i] belongs to the window starting at i
                sequence_length=seqlen,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=2026,
            )

        # Sequence length
        seqlen = self.config['seqlen']

        # Generate train, validation and test sequence data
        self.train_ds = make_windows(scaledtrain, seqlen, self.config['batch_size'], shuffle=True)
        self.val_ds = make_windows(scaledval, seqlen, self.config['batch_size'])
        self.test_ds = make_windows(scaledtest, seqlen, self.config['batch_size'])

        # Extract the very first batch from the dataset
        batch_inputs, batch_targets = next(iter(self.train_ds))

        print("=== First Batch Shapes ===")
        print("Inputs Shape (batch_size, sequence_length, features):", batch_inputs.shape)
        print("Targets Shape (batch_size, target_dim):", batch_targets.shape)

        print("\n=== First Batch Actual Data ===")
        print("Inputs:\n", batch_inputs.numpy())
        print("Targets:\n", batch_targets.numpy())

    # Create a model using the Functional API
    def create_model(self, units, input_shape, activation, lr):

        keras.utils.clear_session()

        model = keras.Sequential([
                keras.Input(shape=input_shape),
                layers.LSTM(units, activation=activation, name='LSTM1', return_sequences=False),
                layers.Dense(1, name='Output'),
            ], name=self.config['model'])
        
                # specify the optimizer separately (preferred method)
        opt = keras.optimizers.Adam(learning_rate=lr)

        # model compilation
        model.compile(optimizer=opt, loss='mse', metrics=['mae'])
        
        return model

    @staticmethod
    def score(y, yhat):
            return {'RMSE': root_mean_squared_error(y, yhat),
                    'MAE': mean_absolute_error(y, yhat),
                    'R²': r2_score(y, yhat)}

    @staticmethod
    def plot_comparison(y_true, y_pred, dates, train_data, lstm_scores):
        # Plot predictions vs actuals over the test block
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(dates, y_true, lw=1.2, label='Actual')
        ax.plot(dates, y_pred, lw=1.2, label='LSTM prediction')
        ax.axhline(train_data['close'].max(), color='tab:red', ls='--', lw=1, label='Training maximum')
        ax.set(title=f"S&P 500 next-day prediction on the test block (R² = {lstm_scores['R²']:.4f})", ylabel='Index level')
        ax.legend()

    def main(self):
        self.load_data()
        model = self.create_model(
            units=self.config['units'],
            input_shape=(self.config['seqlen'], 1),
            activation=self.config['activation'],
            lr=self.config['learning_rate']
        )
        model.summary()
        keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)

        # Callbacks: stop on validation loss, keep the best model, stream metrics to W&B
        model_path = (self.config['results_path'] / 'model.keras').as_posix()

        my_callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.config['patience'],
                                        restore_best_weights=True, verbose=1),
            keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True),
            keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1)
        ]

        history = model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.config['epochs'],
            callbacks=my_callbacks,
            shuffle=False,                # shuffling already happens inside train_ds
            verbose=2,
        )

        # Learning curves (the same curves are live in W&B)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history.history['loss'], label='Train')
        ax.plot(history.history['val_loss'], label='Validation')
        ax.set(title='Learning curves', xlabel='Epoch', ylabel='MSE (scaled, log scale)', yscale='log')
        ax.legend()

        # Load the best checkpoint
        model = keras.models.load_model(model_path)

        # Saved model info
        # model.optimizer.get_config()          # optimizer + learning rate
        # model.loss                            # loss function

        # Evaluate on the test windows (scaled units)
        test_scores = model.evaluate(self.test_ds, verbose=0, return_dict=True)
        print(f"Test MSE (scaled): {test_scores['loss']:.6f} | Test MAE (scaled): {test_scores['mae']:.4f}")

        # Predict on the test set, invert the scaling, and score in index points
        y_pred = self.scaler.inverse_transform(model.predict(self.test_ds, verbose=0)).ravel()
        y_true = self.test_data['close'].to_numpy()[self.config['seqlen']:]
        dates = self.test_data.index[self.config['seqlen']:]

        lstm_scores = self.score(y_true, y_pred)
        print(' | '.join(f'{k}: {v:,.4f}' for k, v in lstm_scores.items()))

        self.plot_comparison(y_true, y_pred, dates, self.train_data, lstm_scores)

    
if __name__ == "__main__":
    config = dict(
    symbol='nifty',             # quantmod bundled dataset: 'spx' or 'nifty'
    start='2012-01-02',         # '2012-01-02' for nifty
    end='2019-12-31',           # '2022-12-30' for nifty
    seqlen=60,                  # lookback window, in trading days
    units=32,                   # LSTM hidden units
    activation='tanh',          # Keras default; see the note in the model section
    batch_size=64,
    epochs=100,
    learning_rate=1e-3,
    patience=15,                # early-stopping patience, in epochs
    train_frac=0.7,
    val_frac=0.1,               # test = the remaining 20%
    model="lstm_functional",
    results_path=Path("data", "results"),
    )
    model = NN(config)
    model.main()
