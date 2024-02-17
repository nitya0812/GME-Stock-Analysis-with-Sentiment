# region Imports
import pandas as pd
import copy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from datetime import timedelta


# endregion Imports


# region ModelClass
# Defining a class for all our model info
class ModelClass:
    EPOCHS = 150  # Number of Epochs to train model
    BATCH_SIZE = 52  # Batch size to feed model

    def __init__(self, append_sentiment_data: bool = False, spike_sentiment: bool = False):
        # Define dataset object
        self.dataset = DatasetClass(append_sentiment_data, spike_sentiment)
        self.predictions = []  # Create a list to hold predictions
        self.predicted_prices = []

        # Define what days to predict
        self.business_days = pd.date_range(
            start=pd.to_datetime(self.dataset.TRAINING_DATA_END_DATE) + timedelta(days=1),
            periods=66, freq='B')
        self.model = self.return_template_model(self.dataset.X_train.shape[1])
        self.model_train()
        self.model_pred()

    # Defining our base LSTM model to use
    @staticmethod
    def return_template_model(in_shape):
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(in_shape, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def model_train(self):
        self.model.fit(self.dataset.X_train, self.dataset.y_train,
                       epochs=self.EPOCHS,
                       batch_size=self.BATCH_SIZE,
                       verbose=0)

    def model_pred(self):
        # Get the last sequence from the training data
        last_sequence = self.dataset.X_train[-1].reshape((1, self.dataset.SEQUENCE_LENGTH, 1))

        # Predict future prices
        for i in range(len(self.business_days)):
            # Get the prediction (scaled value)
            current_prediction = self.model.predict(last_sequence, verbose=0)[0]

            # Append the prediction
            self.predictions.append(current_prediction)

            # Update the sequence
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = current_prediction

        # Inverse transform the predictions to get actual values
        self.predicted_prices = self.dataset.scaler.inverse_transform(np.array(self.predictions).reshape(-1, 1))


# endregion ModelClass


# region DatasetClass
class DatasetClass:
    SEQUENCE_LENGTH = 32  # Sequence length of input data
    TRAINING_DATA_START_DATE = '2021-01-04'  # Starting date of training range
    TRAINING_DATA_END_DATE = '2021-05-28'  # Ending date of training range
    ORIGINAL_COMPOUND_VALS = []  # Original compound sentiment values
    SPIKED_VALS = []  # Spiked compound sentiment values

    def __init__(self, append_sentiment_data: bool, spike_sentiment: bool):
        self.df = self.return_data(append_sentiment_data, spike_sentiment)  # Read dataset

        # Close price extraction
        self.close_prices = self.df['Close'].values.reshape(-1, 1)  # Extract close prices
        self.scaler = MinMaxScaler(feature_range=(0, 3))  # Define a feature scaling object
        self.scaled_close_prices = self.scaler.fit_transform(self.close_prices)  # Scale close prices

        # Define data to send through model for training
        X, y = self.create_sequences(self.scaled_close_prices, self.SEQUENCE_LENGTH)
        train_indices = self.df[self.df['Date'] <= self.TRAINING_DATA_END_DATE].index  # Indices for training data
        self.X_train = X[:train_indices[-1] - self.SEQUENCE_LENGTH]  # Training input
        self.y_train = y[:train_indices[-1] - self.SEQUENCE_LENGTH]  # Ground truth

    def return_data(self, append_sentiment_data, spike_sentiment):
        data_f = pd.read_csv("GME.csv")

        # Ensure the 'Date' column is in datetime format for proper plotting
        data_f['Date'] = pd.to_datetime(data_f['Date'])

        # Drop all data that isn't relavent
        data_f = data_f.drop(data_f[data_f['Date'] <= self.TRAINING_DATA_START_DATE].index)

        # Perform data cleaning
        data_f.sort_values('Date', inplace=True)

        if append_sentiment_data:
            # Reading Sentiment Compound Values
            senti_comp_vals = pd.read_csv("date_compound.csv")

            # Ensure the 'Date' column is in datetime format
            senti_comp_vals['date'] = pd.to_datetime(senti_comp_vals['date'])

            # Sync values to data points in stocks dataset
            syned_comp_vals = []
            for d in data_f['Date']:
                try:
                    syned_comp_vals.append(senti_comp_vals.loc[(senti_comp_vals['date'] == d)].iloc[0]['compound'])
                except IndexError:
                    syned_comp_vals.append(0.0)  # Account for NaN vals

            self.ORIGINAL_COMPOUND_VALS = syned_comp_vals
            # Spike sentiment data
            if spike_sentiment:
                # range of sine values to spike with
                val_range = [-0.5, 0.45]

                # Arange a value for sine for each datapoint in stocks dataframe
                sin_vals = np.arange(val_range[0], val_range[1],
                                     abs(val_range[0] - val_range[1]) / len(senti_comp_vals))

                # Define values to scale each datapoint by
                scaler = np.sin(2 * np.pi * sin_vals).dot(2)
                scaler[scaler < 0] = 0

                # Scale compound sentiment values by the eariler defined values
                syned_comp_vals = [min(1, abs(x) * (scaler[i] + 1)) for i, x in enumerate(syned_comp_vals)]
                self.SPIKED_VALS = syned_comp_vals

            # Insert Sentiment Values into dataframe
            data_f.insert(7, "Sentiment", syned_comp_vals, True)

        return data_f

    def plot_sentiment_values(self, model_name: str = "unnamed_model", show_spiked: bool = False):
        if len(self.ORIGINAL_COMPOUND_VALS) == 0:
            print("\nNo sentiment values appended for model dataset!")
        else:
            plt.figure(figsize=(14, 7))
            plt.plot(self.ORIGINAL_COMPOUND_VALS)
            if show_spiked:
                if len(self.SPIKED_VALS) > 0:
                    plt.plot(self.SPIKED_VALS)
                else:
                    print("\nSentiment values not spiked for this model!")
            plt.savefig(f"{model_name}")

    @staticmethod
    def create_sequences(data, sequence_length):
        xs, ys = [], []
        for i in range(len(data) - sequence_length):
            x = data[i:(i + sequence_length)]
            y = data[i + sequence_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)


# endregion DatasetClass


if __name__ == '__main__':
    print("Please wait for 1-2 minutes for the models to train and print the output\n")
    # Declare Models
    model_base = ModelClass()
    model_wSentiment = ModelClass(append_sentiment_data=True, spike_sentiment=False)
    model_wSentiment_spiked = ModelClass(append_sentiment_data=True, spike_sentiment=True)

    model_wSentiment_spiked.dataset.plot_sentiment_values(model_name='sentiment', show_spiked=True)

    # Define Dataframe with relavent graphs
    predictions_df = pd.DataFrame({
        'Date': model_base.business_days,
        'Predicted_Close': model_base.predicted_prices.flatten(),
        'Predicted_Close_wSentiment': model_wSentiment.predicted_prices.flatten(),
        'Predicted_Close_wSentiment_spiked': model_wSentiment_spiked.predicted_prices.flatten(),
    })

    # Extract ground truth df for predicted values
    ground_truth = model_base.dataset.df.drop(
        model_base.dataset.df[model_base.dataset.df['Date'] <= model_base.dataset.TRAINING_DATA_END_DATE].index)

    # Merge Dataframes
    predictions_df = predictions_df.merge(ground_truth[['Date', 'Close']], on='Date', how='left')
    predictions_df.rename(columns={'Close': 'Actual_Close'}, inplace=True)
    predictions_df = predictions_df.dropna()

    # Calculate and print Losses and accuracy
    def print_metrics(df: pd.DataFrame):
        mse_base = mean_squared_error(y_true=df['Actual_Close'].to_list(),
                                      y_pred=df['Predicted_Close'].to_list())

        mse_wSentiment = mean_squared_error(y_true=df['Actual_Close'].to_list(),
                                            y_pred=df['Predicted_Close_wSentiment'].to_list())

        mse_wSentiment_spiked = mean_squared_error(y_true=df['Actual_Close'].to_list(),
                                                   y_pred=df['Predicted_Close_wSentiment_spiked'].to_list())

        rmse_base = mse_base ** 0.5
        rmse_wSentiment = mse_wSentiment ** 0.5
        rmse_wSentiment_spiked = mse_wSentiment_spiked ** 0.5

        mae_base = mean_absolute_error(y_true=df['Actual_Close'].to_list(),
                                       y_pred=df['Predicted_Close'].to_list())

        mae_wSentiment = mean_absolute_error(y_true=df['Actual_Close'].to_list(),
                                             y_pred=df['Predicted_Close_wSentiment'].to_list())

        mae_wSentiment_spiked = mean_absolute_error(y_true=df['Actual_Close'].to_list(),
                                                    y_pred=df['Predicted_Close_wSentiment_spiked'].to_list())

        metrics = pd.DataFrame([[mse_base, mse_wSentiment, mse_wSentiment_spiked],
                                [rmse_base, rmse_wSentiment, rmse_wSentiment_spiked],
                                [mae_base, mae_wSentiment, mae_wSentiment_spiked]],
                               columns=['Base', 'wSentiment', 'wSpikedSentiment'],
                               index=['MSE', 'RMSE', 'MAE'])
        print(metrics)


    print_metrics(predictions_df)

    # Plot information
    plt.figure(figsize=(14, 7))

    plt.plot(predictions_df['Date'],
             predictions_df['Actual_Close'],
             label='Actual Close',
             color='blue',
             marker='o')

    plt.plot(predictions_df['Date'],
             predictions_df['Predicted_Close'],
             label='Predicted Close',
             color='red',
             marker='x')

    plt.plot(predictions_df['Date'],
             predictions_df['Predicted_Close_wSentiment'],
             label='Predicted Close wSentiment',
             color='green',
             marker='p')

    plt.plot(predictions_df['Date'],
             predictions_df['Predicted_Close_wSentiment_spiked'],
             label='Predicted Close wSentiment spiked',
             color='yellow',
             marker='^')

    # Adding title and labels with font size adjustments
    plt.title('Comparison graph ', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Closing Price', fontsize=14)

    # Rotating date labels for better visibility
    plt.xticks(rotation=45)

    # Adding a legend to distinguish between actual and predicted values
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.savefig("combined")
    plt.show()
