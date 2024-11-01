import pandas as pd
from datetime import datetime
import numpy as np
import logging
import os

# Set up logging configuration for output readability
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define file and processing settings directly in the code for clarity
input_filename = r"C:\Users\digvi\OneDrive\Desktop\ECEN 403\Export_20220111T000000_20220111T235959.csv"
output_filename = input_filename + ".out.csv"  # Name for the output file
duration_user = 300  # Define the duration in seconds for averaging interval
new_input_filename = input_filename + ".new.csv"  # Temporary file for cleaned data
encodings = ["utf-8", "utf-16", "cp1252"]  # List of encodings to try when reading the CSV

def preprocess(file_name, new_file_name, encodings):
    # Preprocesses a CSV file by attempting different encodings until successful
    # Removes double quotes and saves the cleaned data as a new UTF-8 encoded file

    # Validate if the input file exists
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f"{file_name} does not exist or is not a file.")

    # Ensure the input file is a CSV
    if not file_name.endswith('.csv'):
        raise ValueError(f"{file_name} is not a valid CSV file.")

    # Try each encoding to read the original file and write it as a new UTF-8 encoded file
    for encoding in encodings:
        try:
            # Open the original file in read mode and the new file in write mode
            with open(file_name, 'r', encoding=encoding) as f1, open(new_file_name, 'w', encoding="utf-8") as f2:
                for line in f1:
                    line = line.replace('"', "")  # Remove double quotes in each line
                    f2.write(line)  # Write the cleaned line to the new file
            logging.info(f"File {file_name} successfully read with encoding {encoding} and saved as {new_file_name}.")
            return  # If successful, exit the function
        except UnicodeDecodeError:
            logging.warning(f"Failed to read file {file_name} with encoding {encoding}. Trying next encoding.")
        except IOError as e:
            logging.error(f"I/O error occurred: {e}")
            raise  # Re-raise if I/O fails
    raise UnicodeDecodeError(f"Unable to read the file {file_name} with provided encodings.")

def read_csv_with_fallback(filename, encodings):
    # Attempts to read a CSV file using different encodings until successful
    for encoding in encodings:
        try:
            logging.info(f"Trying to read {filename} with encoding {encoding}.")
            return pd.read_csv(filename, encoding=encoding)  # Return the DataFrame if successful
        except UnicodeDecodeError:
            logging.warning(f"Failed to read {filename} with encoding {encoding}.")
        except FileNotFoundError:
            logging.error(f"File not found: {filename}")
            raise  # Re-raise for critical file errors
        except pd.errors.ParserError:
            logging.error(f"Parsing error in {filename}")
            raise  # Raise if CSV structure is broken
    raise UnicodeDecodeError(f"Unable to read the file {filename} with provided encodings.")

def find_and_store_non_numerical(df):
    # Creates a dictionary to store non-numeric values for each column
    non_numeric_data = {col: ['' for _ in range(len(df))] for col in df.columns[1:]}  # Initialize storage for non-numeric values
    # Iterate over the DataFrame columns, starting from the second column (assuming 'timestamp' is first)
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
        return
    
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
        return
    except Exception as e:
        logging.error(f"Failed to process timestamps: {e}")
        return
    
    # Adding the non-numeric values from the original DataFrame to the averaged DataFrame
    for column in df.columns:
        if column.startswith("Non_Numeric_Values_"):
            # Using `first()` here assumes that all non-numeric values in each group are the same
            df_avg[column] = df.groupby('Timestamp')[column].first().reset_index(drop=True)
    
    try:
        # Save the averaged data along with non-numeric values
        df_avg.to_csv(output_filename, index=False)
        logging.info(f"Averaged data saved to {output_filename}.")
    except IOError as e:
        logging.error(f"I/O error when saving file: {e}")

if __name__ == "__main__":
    # Main block to execute preprocessing, averaging, and fault checking
    logging.info("Starting preprocessing...")
    preprocess(input_filename, new_input_filename, encodings)  # Preprocessing the input file

    logging.info("Calculating average data...")
    average_data(new_input_filename, output_filename, duration_user)  # Calculating the average data

    logging.info("Printing faulty entries...")
    try:
        # Read the newly preprocessed file again for final checks
        df_final = read_csv_with_fallback(new_input_filename, encodings)
        print_faulty_entries(df_final)  # Printing any faulty entries found
    except Exception as e:
        logging.error(f"Error while reading or printing faulty entries: {e}")

    print(f"Processed data saved to {output_filename}")  # Prints if

#END
