import os
import pandas as pd
import sklearn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, r2_score

file_path = r"D:\UnzippedMonicoData\FilesWithFailure2022"
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

def create_model(time_steps, num_features):
    # Define Sequential model
    model = Sequential([
        Input(shape=(time_steps, num_features)),
        LSTM(units=128, activation='tanh', return_sequences=True),
        Dropout(0.3),
        LSTM(units=64, activation='tanh'),
        Dropout(0.3),
        Dense(units=1)  
    ])
    
    return model

def run_model() -> str:
    return {"Model Run Output": "TEST"}

def time_till_next_failure(csv_file_path, failure_times_file, scaled_features):
    # Read main CSV into pandas DataFrame
    df = pd.read_csv(csv_file_path, encoding='utf-16', dtype={'Ramsey C4701E.Engine Active Codes': str}, low_memory=False)
    df['Ramsey C4701E.Engine Active Codes'] = df['Ramsey C4701E.Engine Active Codes'].fillna('None')

    # Handle non-numeric values in 'Ramsey C4701E.Fault Relay' by replacing them with NaN, then fill NaNs with 0
    df['Ramsey C4701E.Fault Relay'] = pd.to_numeric(df['Ramsey C4701E.Fault Relay'], errors='coerce').fillna(0)

    # Ensure the timestamp is in datetime format and set it as the index
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)

    # Drop non-numeric columns for averaging
    numeric_col = df.select_dtypes(include='number').columns
    df = df[numeric_col].resample('10s').mean()

    # Apply scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[scaled_features] = df[scaled_features].ffill()
    df[scaled_features] = scaler.fit_transform(df[scaled_features])

    # Load external failure timestamps from the failure times file
    external_failures = pd.read_csv(failure_times_file, parse_dates=['Timestamp'])
    external_failure_times = external_failures['Timestamp'].sort_values()

    # Combine internal and external failure timestamps
    failure_indices = pd.Index(df[df['Ramsey C4701E.Fault Relay'] == 1].index.union(external_failure_times))

    # Initialize a new column for time till next failure
    time_till_failure = [-1.0] * len(df)

    # Loop through the DataFrame to calculate time until the next failure
    for i in range(len(df)):
        # Get the current timestamp
        current_timestamp = df.index[i]
        # Find the next failure time from combined failure indices
        future_failures = failure_indices[failure_indices > current_timestamp]
        if not future_failures.empty:
            # Get the nearest future failure
            next_failure_index = future_failures.min()
            # Calculate the time difference in hours
            time_difference = (next_failure_index - current_timestamp).total_seconds() / 3600
            time_till_failure[i] = time_difference

    # Add the calculated time till next failure as a new column in the DataFrame
    df['time_till_failure'] = time_till_failure

    # Print a sample of the calculated time till failure column to verify results
    print(df[['time_till_failure']].head(20))  # Display the first 20 rows

    return df

def sequence_batch_generator(df, selected_features, target, sequence_length, batch_size):
    num_sequences = len(df) - sequence_length
    num_batches = num_sequences // batch_size
    for batch_idx in range(num_batches):
        sequences = []
        targets = []
        for i in range(batch_size):
            start_idx = batch_idx * batch_size + i
            end_idx = start_idx + sequence_length
            if end_idx < len(df):
                seq = df.iloc[start_idx:end_idx][selected_features].values
                label = df.iloc[end_idx][target].values[0]  
                sequences.append(seq)
                targets.append(label)
        yield np.array(sequences), np.array(targets)

def incremental_training(model, folder_path, selected_features, target, sequence_length, batch_size, save_path="model_Colab_1"):
    # Compile it with a fresh optimizer
    model.compile(optimizer='adam', loss='mean_squared_error')
    # List all CSV files and sort them by date
    csv_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')])
    # Define Early Stopping Callback
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=3,
        restore_best_weights=True
    )
    # Training phase
    print("Starting training phase")
    for csv_file_path in csv_files:
        print(f"Training on file: {csv_file_path}")
        
        # Load and preprocess data
        df = time_till_next_failure(csv_file_path, selected_features)
        
        # Initialize batch generator
        batch_gen = sequence_batch_generator(df, selected_features, target, sequence_length, batch_size)
        
        # Calculate steps per epoch
        steps_per_epoch = (len(df) - sequence_length) // batch_size

        # Incremental training on the batch generator
        model.fit(
            batch_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=1,  
            verbose=1
#            callbacks=[early_stopping],
        )

    # Save the model after training on 2022 data
    model.save(save_path)
    print(f"Training complete! Model saved to {save_path}")

def evaluate_model_on_test_files(model_path, test_folder_path, selected_features, target, sequence_length, batch_size):
    # Load the saved model
    model = load_model(model_path)
    print(f"Loaded model from {model_path}")
    
    # Gather all CSV files from the specified test folder
    test_files = sorted([os.path.join(test_folder_path, f) for f in os.listdir(test_folder_path) if f.endswith('.csv')])

    # Testing phase
    print("\nStarting testing phase...")
    y_true, y_pred = [], []
    for csv_file_path in test_files:
        print(f"Evaluating on file: {csv_file_path}")
        
        # Load and preprocess data
        df = time_till_next_failure(csv_file_path, selected_features)

        # Generate sequences for testing
        batch_gen = sequence_batch_generator(df, selected_features, target, sequence_length, batch_size)
        
        # Predict for each batch
        for batch_idx, (X_batch, y_batch) in enumerate(batch_gen):
            y_pred_batch = model.predict(X_batch)
            y_true.extend(y_batch)
            y_pred.extend(y_pred_batch.flatten())
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches for current file")
    
    # Convert to numpy arrays for metric calculations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    mae = mean_absolute_error(y_true, y_pred)  # Mean Absolute Error
    r2 = r2_score(y_true, y_pred)  # R-squared (coefficient of determination)

    print(f"\nTesting complete! Metrics on test set:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R2 Score): {r2}")
    
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

#target = ['time_till_failure']
#batch_size = 32
#sequence_length = 60
#model = load_model(r"C:\Users\micha\ECEN403\MONICO-X-PredictAI\model_2.h5")
#incremental_training(model, r"D:\UnzippedMonicoData\FilesWithFailure2023", features, target, sequence_length, batch_size, save_path="model_2.h5")  

time_till_next_failure(r"D:\UnzippedMonicoData\2022\Export_20220110T000000_20220110T235959.csv", r"D:\UnzippedMonicoData\failure_times_2022.csv", features)

