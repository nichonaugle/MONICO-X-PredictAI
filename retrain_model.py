
import os
import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


selected_features = limited_features = ['0718.Cylinder_04_Transformer_Secondary_Output', 
                    '0718.Cylinder_03_Transformer_Secondary_Output',
                    '0718.Cylinder_10_Transformer_Secondary_Output',
                    '0718.Cylinder_06_Transformer_Secondary_Output',
                    '0718.Cylinder_08_Transformer_Secondary_Output',
                    '0718.Cylinder_01_Transformer_Secondary_Output',
                    '0718.Compressor_Oil_Pressure',
                    '0718.Cylinder_09_Transformer_Secondary_Output',
                    '0718.Cylinder_05_Transformer_Secondary_Output',
                    '0718.Engine_Speed',
                    '0718.Speed',
                    '0718.Desired_Air_Fuel_Ratio',
                    '0718.Cylinder_07_Transformer_Secondary_Output',
                    '0718.Actual_Air_Fuel_Ratio',
                    '0718.Cylinder_12_Transformer_Secondary_Output',
                    '0718.Cylinder_02_Transformer_Secondary_Output',
                    '0718.Cylinder_11_Transformer_Secondary_Output',
                    '0718.Wastegate_Position_Command',
                    '0718.Fuel_Position_Command',
                    '0718.Eng_Left_Pre-Catalyst_Temperature',
                    '0718.Eng_Left_Post-Catalyst_Temperature',
                    '0718.Eng_Right_Post-Catalyst_Temperature',
                    '0718.Engine_Cylinder_01_Exhaust_Port_Temp',
                    '0718.Eng_Right_Pre-Catalyst_Temperature',
                    '0718.Right_Bank_Exhaust_Port_Temp',
                    '0718.Engine_Average_Exhaust_Port_Temperature',
                    '0718.Gas_Fuel_Flow',
                    '0718.Engine_Cylinder_02_Exhaust_Port_Temp',
                    '0718.Left_Bank_Exhaust_Port_Temp',
                    '0718.Intake_Manifold_Air_Flow',
                    '0718.Engine_Load_Factor',
                    '0718.Air_to_Fuel_Differential_Pressure',
                    '0718.Engine_Cylinder_06_Exhaust_Port_Temp',
                    '0718.Engine_Cylinder_07_Exhaust_Port_Temp',
                    '0718.Engine_Cylinder_10_Exhaust_Port_Temp',
                    '0718.Engine_Cylinder_08_Exhaust_Port_Temp',
                    '0718.Engine_Cylinder_05_Exhaust_Port_Temp',
                    '0718.Engine_Cylinder_03_Exhaust_Port_Temp',
                    '0718.Engine_Cylinder_09_Exhaust_Port_Temp',
                    '0718.1st_Stage_A_Discharge_Pressure',
                    '0718.Actual_Intake_Manifold_Air_Pressure',
                    '0718.Inlet_Manifold_Air_Pressure',
                    '0718.Cylinder_2_Rodload_Tension',
                    '0718.Cylinder_1_Rodload_Tension',
                    '0718.Cylinder_3_Rodload_Tension',
                    '0718.Cylinder_4_Rodload_Tension',
                    '0718.Engine_Cylinder_11_Exhaust_Port_Temp',
                    '0718.Right_Bank_Average_Combustion_Time',
                    '0718.Left_Bank_Average_Combustion_Time',
                    '0718.Cylinder_3_Rodload_Compression',
                    '0718.Cylinder_4_Rodload_Compression',
                    '0718.Cylinder_2_Rodload_Compression',
                    '0718.Cylinder_1_Rodload_Compression',
                    '0718.Frame_Main_Bearing_3_Temperature',
                    '0718.Frame_Main_Bearing_1_Temperature',
                    '0718.Frame_Main_Bearing_2_Temperature',
                    '0718.Engine_Oil_Pressure'
]

def add_time_until_failure_from_folder(folder_path, selected_features, fault_col="Ramsey C4701E.Fault Relay", 
                                       time_col="Timestamp", interval=30):
    """
    Load and average CSV files from a folder, then add a Time_Until_Failure column.

    Args:
        folder_path (str): Folder containing CSV files.
        selected_features (list): List of features to average.
        fault_col (str): Column indicating failure (1 = fault).
        time_col (str): Timestamp column.
        interval (int): Averaging interval in seconds.

    Returns:
        pd.DataFrame: Combined DataFrame with Time_Until_Failure column.
    """
    all_data = pd.DataFrame()
    files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")])

    for file in files:
        df = pd.read_csv(file, encoding='utf-16', low_memory=False)
        if fault_col in df.columns:
            print(f"Processing file: {file}")
            df[time_col] = pd.to_datetime(df[time_col])
            df = df[selected_features + [fault_col, time_col]]
            df.set_index(time_col, inplace=True)

            agg_dict = {col: 'mean' for col in selected_features}
            agg_dict[fault_col] = 'max'

            df_resampled = df.resample(f"{interval}s").agg(agg_dict).reset_index()
            all_data = pd.concat([all_data, df_resampled], ignore_index=True)

    all_data.sort_values(time_col, inplace=True)
    all_data.reset_index(drop=True, inplace=True)

    # Add time until failure
    all_data["Time_Until_Failure"] = np.nan
    failure_indices = all_data[all_data[fault_col] == 1].index
    last_failure_time = None
    for idx in reversed(all_data.index):
        if idx in failure_indices:
            last_failure_time = all_data.loc[idx, time_col]
            all_data.loc[idx, "Time_Until_Failure"] = 0
        elif last_failure_time is not None:
            delta_hours = (last_failure_time - all_data.loc[idx, time_col]).total_seconds() / 3600
            all_data.loc[idx, "Time_Until_Failure"] = delta_hours

    return all_data


def train_random_forest_for_failure(pkl_path, df, selected_features, target_col="Time_Until_Failure"):
    """
    Load a Random Forest model from a .pkl file and retrain it on new data.

    Args:
        df (pd.DataFrame): DataFrame with features and target.
        selected_features (list): Columns to use as features.
        target_col (str): Target column for regression.
        pkl_path (str): Path to the existing model .pkl file.
 
    Returns:
        model (RandomForestRegressor), metrics (dict): Trained model and evaluation metrics.
    """
    df = df.dropna(subset=[target_col])
    df = df.dropna()

    split_index = int(len(df) * 0.8)
    X_train = df[selected_features].iloc[:split_index]
    X_test = df[selected_features].iloc[split_index:]
    y_train = df[target_col].iloc[:split_index]
    y_test = df[target_col].iloc[split_index:]

    model = joblib.load(pkl_path)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    feature_importances = pd.DataFrame({"Feature": selected_features, "Importance": model.feature_importances_})
    feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {"MAE": mae, "R² Score": r2}
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R² Score: {r2}")
    return model, metrics
