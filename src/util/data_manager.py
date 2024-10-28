import pandas as pd
from datetime import datetime
import numpy as np
import logging
import os

# setting up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def preprocess(file_name, new_file_name, encodings):
    # try each encoding to read the original file and write it as a new utf-8 encoded file
    
    # validate the input filename
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f"{file_name} does not exist or is not a file.")
    
    if not file_name.endswith('.csv'):
        raise ValueError(f"{file_name} is not a valid CSV file.")
    
    
    for encoding in encodings:
        try:
            # open the original file in read mode and the new file in write mode
            with open(file_name, 'r', encoding=encoding) as f1, open(new_file_name, 'w', encoding="utf-8") as f2:
                for line in f1:
                    line = line.replace('"', "")  # remove double quotes in each line
                    f2.write(line)  # writing the new line to the new file
            logging.info(f"file {file_name} successfully read with encoding {encoding} and saved as {new_file_name}.")
            return  # if successful, exit the function
        except UnicodeDecodeError:
            logging.warning(f"failed to read file {file_name} with encoding {encoding}. trying next encoding.")
        except IOError as e:
            logging.error(f"i/o error occurred: {e}")
            raise  # re-raise if i/o fails
    raise UnicodeDecodeError(f"unable to read the file {file_name} with provided encodings.")

def read_csv_with_fallback(filename, encodings):
    # to read the csv using different encodings until successful
    for encoding in encodings:
        try:
            logging.info(f"trying to read {filename} with encoding {encoding}.")
            return pd.read_csv(filename, encoding=encoding)
        except UnicodeDecodeError:
            logging.warning(f"failed to read {filename} with encoding {encoding}.")
        except FileNotFoundError:
            logging.error(f"file not found: {filename}")
            raise  # re-raise for critical file errors
        except pd.errors.ParserError:
            logging.error(f"parsing error in {filename}")
            raise  # raise if csv structure is broken
    raise UnicodeDecodeError(f"unable to read the file {filename} with provided encodings.")

def find_and_store_non_numerical(df):
    # creating a dictionary to store non-numeric values for each column
    non_numeric_data = {col: ['' for _ in range(len(df))] for col in df.columns[1:]}
    # iterating over the dataframe columns, starting from the second column (assuming 'timestamp' is first)
    for column in df.columns[1:]:
        try:
            # converting each column to numeric, forcing errors to nan
            non_numeric = pd.to_numeric(df[column], errors='coerce')
            # getting the indices where the non-numeric entries are found
            non_numeric_indices = non_numeric[non_numeric.isna()].index
            # for each index with non-numeric values, store the value to the corresponding position in non_numeric_data
            for idx in non_numeric_indices:
                non_numeric_data[column][idx] = df.loc[idx, column]
            # replacing the non-numeric values in the original column with nan
            df[column] = non_numeric
        except Exception as e:
            logging.warning(f"failed to process column {column}: {e}")
    # creating new columns in the dataframe for each column that had non-numeric values
    for column, values in non_numeric_data.items():
        if any(values):  # only add the column if it contains non-numeric data
            df[f'Non_Numeric_Values_{column}'] = values
    logging.info("non-numeric values identified and stored in new columns.")
    return df

def print_faulty_entries(df):
    # checks for faulty machines indicated by '1' in the ramsey c4701e.fault relay column
    if 'Ramsey C4701E.Fault Relay' not in df.columns:
        logging.error("column 'ramsey c4701e.fault relay' not found in the dataframe.")
        return  # exit if column is not found
    # check for rows with a '1' in the fault relay column
    faulty_entries = df[df['Ramsey C4701E.Fault Relay'] == 1]
    # if there are faulty entries found, that is '1' which means that the machine has failed
    if not faulty_entries.empty:
        print("When fault relay is 1 (component has failed):")
        for index, row in faulty_entries.iterrows():
            try:
                # getting the timestamp
                timestamp = row['Timestamp']
                # getting non-numeric values for that row
                non_numeric_values = {col: row[col] for col in df.columns if pd.isna(pd.to_numeric(row[col], errors='coerce'))}
                # printing the timestamp and only the non-numeric values
                if non_numeric_values:  # only print if there are non-numeric values
                    print(f"timestamp: {timestamp}, non-numeric values: {non_numeric_values}")
                else:
                    print(f"timestamp: {timestamp}, no non-numeric values found.")
            except Exception as e:
                logging.warning(f"error while processing faulty entry at index {index}: {e}")
    else:
        print("no faulty entries found in 'ramsey c4701e.fault relay'.")

def average_data(input_filename, output_filename, duration_user):
    encodings = ["utf-8", "utf-16", "cp1252"]
    try:
        df = read_csv_with_fallback(input_filename, encodings)
    except Exception as e:
        logging.error(f"failed to load csv file: {e}")
        return
    
    # finding non-numeric values and store them in new columns
    df = find_and_store_non_numerical(df)
    try:
        # taking in timestamp and calculate averages
        df['Timestamp'] = df['Timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').timestamp())  # converting the timestamp into a number
        df['Timestamp'] = (df['Timestamp'] // duration_user) * duration_user  # round it down to the nearest multiple of duration_user
        df_avg = df.groupby('Timestamp', as_index=False).mean()  # calculating the mean for numeric columns
        # converting the timestamp back to a readable format
        df_avg['Timestamp'] = df_avg['Timestamp'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%dT%H:%M:%S'))
    except Exception as e:
        logging.error(f"failed to process timestamps: {e}")
        return
    
    # adding the non_numeric_values from the original dataframe to the averaged dataframe
    for column in df.columns:
        if column.startswith("Non_Numeric_Values_"):
            # using `first()` here assumes that all non-numeric values in each group are the same
            df_avg[column] = df.groupby('Timestamp')[column].first().reset_index(drop=True)
    
    try:
        # saving the averaged data along with non-numeric values
        df_avg.to_csv(output_filename, index=False)
        logging.info(f"averaged data saved to {output_filename}.")
    except IOError as e:
        logging.error(f"i/o error when saving file: {e}")

if __name__ == "__main__":
    input_filename = r"C:\Users\digvi\OneDrive\Desktop\ECEN 403\Export_20220111T000000_20220111T235959.csv"
    output_filename = input_filename + ".out.csv"
    duration_user = 300  # duration in seconds can be changed
    new_input_filename = input_filename + ".new.csv"  # creates a new csv file with actual columns unlike the original csv file
    encodings = ["utf-8", "utf-16", "cp1252"]

    logging.info("starting preprocessing...")
    preprocess(input_filename, new_input_filename, encodings)  # preprocessing the input file

    logging.info("calculating average data...")
    average_data(new_input_filename, output_filename, duration_user)  # calculating the average data

    logging.info("printing faulty entries...")
    try:
        df_final = read_csv_with_fallback(new_input_filename, encodings)  # reads the newly preprocessed file again for final checks
        print_faulty_entries(df_final)  # printing any faulty entries found
    except Exception as e:
        logging.error(f"error while reading or printing faulty entries: {e}")

    print(f"processed data saved to {output_filename}")  # prints if everything is successful
    logging.info("process completed successfully.")
