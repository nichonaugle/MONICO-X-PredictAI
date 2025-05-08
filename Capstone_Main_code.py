#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from datetime import datetime
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging configuration for output readability
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define file and processing settings directly in the code for clarity
input_filename = r"C:\Users\digvi\OneDrive\Desktop\ECEN 403\Export_20230106T000000_20230106T235959.csv"
output_filename = input_filename + ".out.csv"  # Name for the output file
duration_user = 300  # Define the duration in seconds for averaging interval
new_input_filename = input_filename + ".new.csv"  # Temporary file for cleaned data
encodings = ["utf-8", "utf-16", "cp1252"]  # List of encodings to try when reading the CSV

def preprocess(file_name, new_file_name, encodings):
    # Preprocesses a CSV file by attempting different encodings until successful
    # Removes double quotes and saves the cleaned data as a new UTF-8 encoded file
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f"{file_name} does not exist or is not a file.")
    if not file_name.endswith('.csv'):
        raise ValueError(f"{file_name} is not a valid CSV file.")
    for encoding in encodings:
        try:
            with open(file_name, 'r', encoding=encoding) as f1, open(new_file_name, 'w', encoding="utf-8") as f2:
                for line in f1:
                    line = line.replace('"', "")  # Remove double quotes in each line
                    f2.write(line)
            logging.info(f"File {file_name} successfully read with encoding {encoding} and saved as {new_file_name}.")
            return
        except UnicodeDecodeError:
            logging.warning(f"Failed to read file {file_name} with encoding {encoding}. Trying next encoding.")
        except IOError as e:
            logging.error(f"I/O error occurred: {e}")
            raise
    raise UnicodeDecodeError(f"Unable to read the file {file_name} with provided encodings.")

def read_csv_with_fallback(filename, encodings):
    # Attempts to read a CSV file using different encodings until successful
    for encoding in encodings:
        try:
            logging.info(f"Trying to read {filename} with encoding {encoding}.")
            return pd.read_csv(filename, encoding=encoding)
        except UnicodeDecodeError:
            logging.warning(f"Failed to read {filename} with encoding {encoding}.")
        except FileNotFoundError:
            logging.error(f"File not found: {filename}")
            raise
        except pd.errors.ParserError:
            logging.error(f"Parsing error in {filename}")
            raise
    raise UnicodeDecodeError(f"Unable to read the file {filename} with provided encodings.")

def find_and_store_non_numerical(df):
    # Creates a dictionary to store non-numeric values for each column
    non_numeric_data = {col: ['' for _ in range(len(df))] for col in df.columns[1:]}  # Initialize storage for non-numeric values
    # Iterate over the DataFrame columns, starting from the second column (assuming 'Timestamp' is first)
    for column in df.columns[1:]:
        try:
            # Convert each column to numeric, forcing errors to NaN
            non_numeric = pd.to_numeric(df[column], errors='coerce')
            # Get the indices where non-numeric entries are found
            non_numeric_indices = non_numeric[non_numeric.isna()].index
            # For each index with non-numeric values, store the value to the corresponding position in non_numeric_data
            for idx in non_numeric_indices:
                non_numeric_data[column][idx] = df.loc[idx, column]
            # Replace the non-numeric values in the original column with NaN
            df[column] = non_numeric
        except Exception as e:
            logging.warning(f"Failed to process column {column}: {e}")
    
    # Creating new columns in the DataFrame for each column that had non-numeric values
    for column, values in non_numeric_data.items():
        if any(values):  # Only add the column if it contains non-numeric data
            df[f'Non_Numeric_Values_{column}'] = values
    logging.info("Non-numeric values identified and stored in new columns.")
    return df

def print_faulty_entries(df):
    # Checks for faulty machines indicated by '1' in the 'Ramsey C4701E.Fault Relay' column
    if 'Ramsey C4701E.Fault Relay' not in df.columns:
        logging.error("Column 'Ramsey C4701E.Fault Relay' not found in the DataFrame.")
        return  # Exit if column is not found
    # Check for rows with a '1' in the fault relay column
    faulty_entries = df[df['Ramsey C4701E.Fault Relay'] == 1]
    # If there are faulty entries found, which indicates that the machine has failed
    if not faulty_entries.empty:
        print("When fault relay is 1 (component has failed):")
        for index, row in faulty_entries.iterrows():
            try:
                # Get the timestamp
                timestamp = row['Timestamp']
                # Get non-numeric values for that row
                non_numeric_values = {col: row[col] for col in df.columns if pd.isna(pd.to_numeric(row[col], errors='coerce'))}
                # Print the timestamp and only the non-numeric values
                if non_numeric_values:  # Only print if there are non-numeric values
                    print(f"Timestamp: {timestamp}, Non-numeric values: {non_numeric_values}")
                else:
                    print(f"Timestamp: {timestamp}, no non-numeric values found.")
            except Exception as e:
                logging.warning(f"Error while processing faulty entry at index {index}: {e}")
    else:
        print("No faulty entries found in 'Ramsey C4701E.Fault Relay'.")

def average_data(input_filename, output_filename, duration_user):
    # Function to average data in the DataFrame based on a specified duration
    encodings = ["utf-8", "utf-16", "cp1252"]  # List of encodings to try for reading the file
    try:
        # Attempt to read the CSV file with fallback encodings
        df = read_csv_with_fallback(input_filename, encodings)
    except Exception as e:
        logging.error(f"Failed to load CSV file: {e}")
        return None  # Return None if loading fails
    
    # Finding non-numeric values and store them in new columns
    df = find_and_store_non_numerical(df)
    try:
        # Convert the 'Timestamp' column into a numeric format
        df['Timestamp'] = df['Timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').timestamp())
        
        # Check if duration_user is valid (greater than zero)
        if duration_user <= 0:
            raise ValueError("Duration for averaging must be greater than zero.")
        
        # Round down to the nearest multiple of duration_user
        df['Timestamp'] = (df['Timestamp'] // duration_user) * duration_user
        df_avg = df.groupby('Timestamp', as_index=False).mean()  # Calculate the mean for numeric columns
        # Convert the timestamp back to a readable format
        df_avg['Timestamp'] = df_avg['Timestamp'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%dT%H:%M:%S'))
    except ValueError as e:
        logging.error(f"Value error: {e}")
        return None
    except Exception as e:
        logging.error(f"Failed to process timestamps: {e}")
        return None
    
    # Adding the non-numeric values from the original DataFrame to the averaged DataFrame
    for column in df.columns:
        if column.startswith("Non_Numeric_Values_"):
            df_avg[column] = df.groupby('Timestamp')[column].first().reset_index(drop=True)
    
    try:
        # Save the averaged data along with non-numeric values
        df_avg.to_csv(output_filename, index=False)
        logging.info(f"Averaged data saved to {output_filename}.")
    except IOError as e:
        logging.error(f"I/O error when saving file: {e}")
    
    return df_avg  # Return the averaged DataFrame for further analysis

# New function to store data in array format
def store_data_in_array(output_filename):
    # Function to read the averaged data from the output CSV and store it in an array format
    try:
        # Read the averaged CSV file into a DataFrame
        df = pd.read_csv(output_filename)
        
        # Convert each row in the DataFrame to a list (array) and store them in a list
        data_as_array = df.values.tolist()
        
        logging.info(f"Data from {output_filename} successfully converted to array format.")
        
        return data_as_array
    
    except Exception as e:
        logging.error(f"Error occurred while converting data to array format: {e}")
        return None

def data_analysis(df):
    # Data Cleaning and Preprocessing Section
    # Handle non-numeric data: Use categories for missing data and sensor issues
    # Replace "No Data" with 'Missing' and "None" with 'Malfunction' for clarity
    df.replace('No Data', 'Missing', inplace=True)
    df.replace('None', 'Malfunction', inplace=True)
    
    # Replace these categories with NaN to allow numeric analysis where needed
    df.replace(['Missing', 'Malfunction'], np.nan, inplace=True)
    
    # Outlier Detection for Key Metrics
    
    # Key metrics for outlier analysis - identifies potential issues in system performance
    key_metrics = ['0718.1st_Stage_A_Discharge_Pressure', '0718.Actual_Air_Fuel_Ratio', '0718.System_Battery_Voltage']
    
    # Calculate IQR-based outliers to flag extreme values
    outliers = {}
    for metric in key_metrics:
        if metric in df.columns:
            metric_data = df[metric].dropna()
            q1 = metric_data.quantile(0.25)
            q3 = metric_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers[metric] = metric_data[(metric_data < lower_bound) | (metric_data > upper_bound)]
        else:
            logging.warning(f"Metric {metric} not found in DataFrame columns.")
    
    # Visualize outliers using boxplots to see the spread of data points
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[key_metrics])
    plt.title("Outlier Detection for Key Metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.show()
    
    # Statistical Analysis for Detailed Breakdown
    
    # Summary statistics for each sensor to show average, median, and spread
    # Helps in understanding central tendency and variability in the readings
    stats_summary = df.describe()
    print(stats_summary)
    
    # Moving averages for trend analysis to spot seasonal patterns or shifts
    for metric in key_metrics:
        if metric in df.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(df['Timestamp'], df[metric], label='Original Data')
            plt.plot(df['Timestamp'], df[metric].rolling(window=10).mean(), label='10-point Moving Average', color='orange')
            plt.xticks(rotation=45)
            plt.legend(loc="best")
            plt.title(f"Trend Analysis for {metric}")
            plt.xlabel("Timestamp")
            plt.ylabel(metric)
            plt.show()
        else:
            logging.warning(f"Metric {metric} not found in DataFrame columns.")
    
    # Correlation Analysis to Reveal Relationships Between Sensors
    
    # Generate correlation matrix to explore variable relationships
    # Useful for identifying dependencies or redundancies among metrics
    numeric_cols = df.select_dtypes(include=[np.float64, np.int64]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    # Heatmap of correlation matrix for visual inspection of sensor relationships
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title("Correlation Matrix of Sensor Readings")
    plt.show()
    
    # Extract highly correlated pairs (> 0.7) to examine interdependencies
    high_corr_vars = correlation_matrix[correlation_matrix.abs() > 0.7].stack().reset_index()
    high_corr_vars = high_corr_vars[high_corr_vars['level_0'] != high_corr_vars['level_1']]
    high_corr_vars.columns = ['Variable 1', 'Variable 2', 'Correlation']
    high_corr_vars.drop_duplicates(inplace=True)
    
    # Display the high correlation pairs for further analysis
    print("Highly correlated variable pairs:")
    print(high_corr_vars)

if __name__ == "__main__":
    logging.info("Starting preprocessing...")
    preprocess(input_filename, new_input_filename, encodings)

    logging.info("Calculating average data...")
    df_avg = average_data(new_input_filename, output_filename, duration_user)

    if df_avg is not None:
        logging.info("Printing faulty entries...")
        try:
            df_final = read_csv_with_fallback(new_input_filename, encodings)
            print_faulty_entries(df_final)
        except Exception as e:
            logging.error(f"Error while reading or printing faulty entries: {e}")
        
        logging.info("Performing data analysis on the averaged data...")
        data_analysis(df_avg)
        
        # Convert averaged data to array format and print or use
        data_array = store_data_in_array(output_filename)
        if data_array is not None:
            print("Averaged Data in Array Format:")
            print(data_array)
    else:
        logging.error("Data averaging failed. Skipping data analysis.")

    print(f"Processed data saved to {output_filename}")  # Notify user of saved processed data


# In[ ]:




