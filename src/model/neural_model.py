import os
import pandas as pd
import sklearn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

csv_directory = Path(__file__).parent / "data"  # Assuming 'data' is a folder in your project
file_path = csv_directory / "test_data.csv"

features = ['0718.1st_Stage_A_Discharge_Pressure','0718.1st_Stage_A_Suction_Pressure','0718.Acceleration_Ramp_Rate','0718.Actual_Air_Fuel_Ratio',
                   '0718.Actual_Engine_Timing','0718.Actual_Intake_Manifold_Air_Pressure','0718.Air_to_Fuel_Differential_Pressure','0718.Average_Combustion_Time',
                   '0718.Choke_Compensation_Percentage','0718.Choke_Gain_Percentage','0718.Choke_Position_Command','0718.Choke_Stability_Percentage','0718.Compressor_Oil_Pressure',               
                   '0718.Compressor_Oil_Temperature','0718.Crankcase_Air_Pressure','0718.Crank_Terminate_Speed_Setpoint','0718.Cylinder_01_Detonation_Level','0718.Cylinder_01_Filtered_Combustion_Time',
                   '0718.Cylinder_01_Ignition_Timing','0718.Cylinder_01_Transformer_Secondary_Output','0718.Cylinder_01_Unfiltered_Combustion_Time','0718.Cylinder_02_Detonation_Level','0718.Cylinder_02_Filtered_Combustion_Time',
                   '0718.Cylinder_02_Ignition_Timing','0718.Cylinder_02_Transformer_Secondary_Output','0718.Cylinder_02_Unfiltered_Combustion_Time','0718.Cylinder_03_Detonation_Level','0718.Cylinder_03_Filtered_Combustion_Time',
                   '0718.Cylinder_03_Ignition_Timing','0718.Cylinder_03_Transformer_Secondary_Output','0718.Cylinder_03_Unfiltered_Combustion_Time','0718.Cylinder_04_Detonation_Level','0718.Cylinder_04_Filtered_Combustion_Time',
                   '0718.Cylinder_04_Ignition_Timing','0718.Cylinder_04_Transformer_Secondary_Output','0718.Cylinder_04_Unfiltered_Combustion_Time','0718.Cylinder_05_Detonation_Level','0718.Cylinder_05_Filtered_Combustion_Time',
                   '0718.Cylinder_05_Ignition_Timing','0718.Cylinder_05_Transformer_Secondary_Output','0718.Cylinder_05_Unfiltered_Combustion_Time','0718.Cylinder_06_Detonation_Level','0718.Cylinder_06_Filtered_Combustion_Time',
                   '0718.Cylinder_06_Ignition_Timing','0718.Cylinder_06_Transformer_Secondary_Output','0718.Cylinder_06_Unfiltered_Combustion_Time','0718.Cylinder_07_Detonation_Level','0718.Cylinder_07_Filtered_Combustion_Time',
                   '0718.Cylinder_07_Ignition_Timing','0718.Cylinder_07_Transformer_Secondary_Output','0718.Cylinder_07_Unfiltered_Combustion_Time','0718.Cylinder_08_Detonation_Level','0718.Cylinder_08_Filtered_Combustion_Time',
                   '0718.Cylinder_08_Ignition_Timing','0718.Cylinder_08_Transformer_Secondary_Output','0718.Cylinder_08_Unfiltered_Combustion_Time','0718.Cylinder_09_Detonation_Level','0718.Cylinder_09_Filtered_Combustion_Time',
                   '0718.Cylinder_09_Ignition_Timing','0718.Cylinder_09_Transformer_Secondary_Output','0718.Cylinder_09_Unfiltered_Combustion_Time','0718.Cylinder_10_Detonation_Level','0718.Cylinder_10_Filtered_Combustion_Time',
                   '0718.Cylinder_10_Ignition_Timing','0718.Cylinder_10_Transformer_Secondary_Output','0718.Cylinder_10_Unfiltered_Combustion_Time','0718.Cylinder_11_Detonation_Level','0718.Cylinder_11_Filtered_Combustion_Time',
                   '0718.Cylinder_11_Ignition_Timing','0718.Cylinder_11_Transformer_Secondary_Output','0718.Cylinder_11_Unfiltered_Combustion_Time','0718.Cylinder_12_Detonation_Level','0718.Cylinder_12_Filtered_Combustion_Time',
                   '0718.Cylinder_12_Ignition_Timing','0718.Cylinder_12_Transformer_Secondary_Output','0718.Cylinder_12_Unfiltered_Combustion_Time',
                   '0718.Cylinder_1_A_Discharge_Temperature','0718.Cylinder_1_Rodload_Compression','0718.Cylinder_1_Rodload_Tension',
                   '0718.Cylinder_2_A_Discharge_Temperature','0718.Cylinder_2_Rodload_Compression','0718.Cylinder_2_Rodload_Tension',
                   '0718.Cylinder_3_A_Discharge_Temperature','0718.Cylinder_3_Rodload_Compression','0718.Cylinder_3_Rodload_Tension',
                   '0718.Cylinder_4_A_Discharge_Temperature','0718.Cylinder_4_Rodload_Compression','0718.Cylinder_4_Rodload_Tension',
                   '0718.Desired_Air_Fuel_Ratio','0718.Desired_Combustion_Time','0718.Desired_Engine_Exhaust_Port_Temperature','0718.Desired_Engine_Speed','0718.Desired_Intake_Manifold_Air_Pressure','0718.Engine_Average_Exhaust_Port_Temperature',
                   '0718.Engine_Coolant_Pressure','0718.Engine_Coolant_Temperature','0718.Engine_Cylinder_01_Exhaust_Port_Temp','0718.Engine_Cylinder_02_Exhaust_Port_Temp','0718.Engine_Cylinder_03_Exhaust_Port_Temp',
                   '0718.Engine_Cylinder_04_Exhaust_Port_Temp','0718.Engine_Cylinder_05_Exhaust_Port_Temp','0718.Engine_Cylinder_06_Exhaust_Port_Temp','0718.Engine_Cylinder_07_Exhaust_Port_Temp','0718.Engine_Cylinder_08_Exhaust_Port_Temp','0718.Engine_Cylinder_09_Exhaust_Port_Temp',
                   '0718.Engine_Cylinder_10_Exhaust_Port_Temp','0718.Engine_Cylinder_11_Exhaust_Port_Temp','0718.Engine_Cylinder_12_Exhaust_Port_Temp','0718.Engine_Load_Factor','0718.Engine_Oil_Filter_Differential_Pressure','0718.Engine_Oil_Pressure',
                   '0718.Engine_Oil_Temperature','0718.Engine_Oil_to_Engine_Coolant_Differential_Temperature','0718.Engine_Overcrank_Time','0718.Engine_Prelube_Time_Out_Period','0718.Engine_Purge_Cycle_Time','0718.Engine_Speed','0718.Eng_Left_Catalyst_Differential_Pressure',
                   '0718.Eng_Left_Post-Catalyst_Temperature','0718.Eng_Left_Pre-Catalyst_Temperature','0718.Eng_Right_Catalyst_Differential_Pressure','0718.Eng_Right_Post-Catalyst_Temperature','0718.Eng_Right_Pre-Catalyst_Temperature','0718.First_Desired_Timing',
                   '0718.Frame_Main_Bearing_1_Temperature','0718.Frame_Main_Bearing_2_Temperature','0718.Frame_Main_Bearing_3_Temperature','0718.Frame_Main_Bearing_4_Temperature','0718.Fuel_Position_Command','0718.Fuel_Quality','0718.Fuel_Temperature','0718.Gas_Fuel_Correction_Factor',
                   '0718.Gas_Fuel_Flow','0718.Gas_Specific_Gravity','0718.Governor_Compensation_Percentage','0718.Governor_Gain_Percentage','0718.Governor_Stability_Percentage','0718.Inlet_Manifold_Air_Pressure','0718.Intake_Manifold_Air_Flow','0718.Intake_Manifold_Air_Temperature',
                   '0718.Left_Bank_Average_Combustion_Time','0718.Left_Bank_Exhaust_Port_Temp','0718.Left_Bank_Turbine_Inlet_Temp','0718.Left_Bank_Turbine_Outlet_Temp','0718.Low_Idle_Speed','0718.Maximum_Choke_Position','0718.Maximum_Engine_High_Idle_Speed','0718.mCore_Heartbeat','0718.Minimum_High_Engine_Idle_Speed',
                   '0718.Right_Bank_Average_Combustion_Time','0718.Right_Bank_Exhaust_Port_Temp','0718.Right_Bank_Turbine_Inlet_Temp','0718.Right_Bank_Turbine_Outlet_Temp','0718.Second_Desired_Timing','0718.Speed','0718.System_Battery_Voltage','0718.Total_Crank_Cycle_Time','0718.Total_Operating_Hours',
                   '0718.Unfiltered_Engine_Oil_Pressure','0718.Wastegate_Compensation_Percentage','0718.Wastegate_Gain_Percentage','0718.Wastegate_Position_Command','0718.Wastegate_Stability_Percentage', '0718.Controller_Operating_Hours'
                  ]

def init() -> None:
    global model
    model = load_model(os.getcwd() + "/src/model/trained/prediction_model.keras", compile = True)

def create_model(time_steps, num_features):
    # Define Sequential model
    model = Sequential([
        LSTM(units=128, activation='tanh', input_shape=(time_steps, num_features), return_sequences=True),
        Dropout(0.4),
        LSTM(units=64, activation='tanh'),
        Dropout(0.4),
        Dense(units=1)  
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Return model
    return model

def run_model(data) -> str:
    return model.predict(data)
    #return {"Model Run Output": "TEST"}

def time_till_next_failure(csv_file_path, scaled_features):
    # Read csv into pandas data frame and fix the Active Codes dtype
    df = pd.read_csv(csv_file_path, encoding = 'utf-16', dtype = {'Ramsey C4701E.Engine Active Codes': str})
    df['Ramsey C4701E.Engine Active Codes'] = df['Ramsey C4701E.Engine Active Codes'].fillna('None')
    # Ensure the timestamp is in a datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    # Drop non numeric columns for averaging
    numeric_col = df.select_dtypes(include='number').columns
    df = df[numeric_col].resample('10s').mean()
    # Apply scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[scaled_features] = df[scaled_features].ffill()
    df[scaled_features] = scaler.fit_transform(df[scaled_features])
    # Initialize a new column for time till next failure
    time_till_failure = [-1.0] * len(df)
    # Find indices where failure occurs
    failure_indices = df[df['Ramsey C4701E.Fault Relay'] == 1].index
    # Loop through data frame
    for i in range(len(df)):
        # Get timestamp of current row 
        current_timestamp = df.index[i]
        # All future failures after the the rows index
        future_failures = failure_indices[failure_indices > current_timestamp]
        if not future_failures.empty:
            # Find the next failure
            next_failure_index = future_failures.min()
            # Caluclate the difference in hours
            time_difference = (next_failure_index - current_timestamp).total_seconds() / 3600
            time_till_failure[i] = time_difference
          
    # Use head to display results to verify
    df = pd.concat([df, pd.Series(time_till_failure, index=df.index, name='time_till_failure')], axis=1)
    print(df[[ 'time_till_failure', '0718.1st_Stage_A_Discharge_Pressure']].head(20))
    return df

def sequence_batch_generator(df, selected_features, target, sequence_length, batch_size):
    # Calculate total number of batches
    num_batches = (len(df) - sequence_length) // (batch_size)
    while True:
    # Loop over batches
        for batch_idx in range(num_batches):
            sequences = []
            targets = []
        # Generate sequences for each batch
        for i in range(batch_size):
            start_idx = batch_idx * batch_size + i
            end_idx = start_idx + sequence_length
            # Check for valid range and append sequence and tagrgets to list
            if end_idx < len(df):
                seq = df.iloc[start_idx:end_idx][selected_features].values
                label = df.iloc[end_idx][target].values[0]  
                sequences.append(seq)
                targets.append(label)
            yield np.array(sequences), np.array(targets)

def train_on_file(model, csv_file_path, selected_features, target, sequence_length, batch_size):
   # Load data
    df = time_till_next_failure(csv_file_path, selected_features)

    # Generate batches using the generator
    batch_gen = sequence_batch_generator(df, selected_features, target, sequence_length, batch_size)
    steps_per_epoch = (len(df) - sequence_length) // (batch_size)

    # Train the model on this file's data
    model.fit(
        batch_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=1,  # Use only one epoch per file for incremental training
        verbose=1,
        reset_metrics=False  # Keep metrics from previous training
    )

def incremental_training(model, folder_path, selected_features, target, sequence_length=50, batch_size=32):
    # Find all CSV files in the folder
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Train model on each file incrementally
    for csv_file_path in csv_files:
        print(f"Training on file: {csv_file_path}")
        train_on_file(model, csv_file_path, selected_features, target, sequence_length, batch_size)

    print("Training complete!")


'''
df = time_till_next_failure(file_path, features)
# Example usage of the generator:
target = ['time_till_failure']
batch_size = 32
sequence_length = 60
steps_per_epoch = (len(df) - sequence_length ) // (batch_size)
batch_generator = sequence_batch_generator(df, features, target, sequence_length, batch_size)

model = create_model(sequence_length, len(features))
model.compile(optimizer='adam', loss='mean_squared_error')
# Print the model architecture
X_batch, y_batch = next(batch_generator)
print("X_batch shape:", X_batch.shape)
print("y_batch shape:", y_batch.shape)
print("Sample y_batch values:", y_batch[:10])

model.summary()
model.fit(batch_generator, 
          steps_per_epoch=steps_per_epoch, 
          epochs=10)  # Adjust the number of epochs as needed
'''