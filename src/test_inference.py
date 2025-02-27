import numpy as np
import os
import asyncio
#from tensorflow.keras.models import load_model

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
limited_features = [ 
    '0718.Cylinder_04_Transformer_Secondary_Output', '0718.Cylinder_03_Transformer_Secondary_Output',
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

"""
# Assuming you have already trained and saved your model
model = load_model(os.getcwd() + "/src/model/trained/prediction_model.keras")

# Example: Assume you have 100 samples, each with 186 features
samples = 1
features = 162

# Generate random data for demonstration
data = np.random.rand(samples, features).astype('float32')  # Shape: (100, 186)

input_data = np.reshape(data, (samples, 1, features))

# Input the data into the model
predictions = model.predict(input_data)
"""
from util.data_collector import DataCollector

dc = DataCollector()
asyncio.run(dc.start())