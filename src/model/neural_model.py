import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# matrices is row x column
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# 1 Preprocess your data using a pandas DataFrame.
# 2 Convert the DataFrame to a numpy array once the data is ready for training.
# 3 Feed the numpy array to the TensorFlow model for training



def my_model(time_steps, num_features):
    # Define the Sequential model
    model = Sequential([
        LSTM(units=64, activation='tanh', input_shape=(time_steps, num_features), return_sequences=True),
        Dropout(0.2),
        LSTM(units=32, activation='tanh'),
        Dropout(0.2),
        Dense(units=1)  # Output layer for regression
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Return the model
    return model

def add_time_till_failure(csv_file_path, save_to_csv=False, output_file='data_with_time_till_failure.csv'):
    df = pd.read_csv(csv_file_path)
    # Ensure the timestamp is in a datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Initialize a new column for time till next failure
    df['time_till_failure'] = None
    # Find indices where failure occurs
    failure_indices = df[df['failure'] == 1].index
    # Loop through the dataframe to fill in the time till next failure
    for i in range(len(df)):
        future_failures = failure_indices[failure_indices > i]
        if not future_failures.empty:
            next_failure_index = future_failures.min()
            time_difference = (df.loc[next_failure_index, 'timestamp'] - df.loc[i, 'timestamp']).total_seconds() / 3600  
            df.loc[i, 'time_till_failure'] = time_difference
        else:
            df.loc[i, 'time_till_failure'] = None  
    # Placeholder value
    df['time_till_failure'].fillna(-1, inplace=True)

    # Save the modified DataFrame back to CSV
    if save_to_csv:
        df.to_csv(output_file, index=False)

    return df




