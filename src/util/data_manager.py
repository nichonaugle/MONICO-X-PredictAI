import pandas as pd
import sys
from datetime import datetime

def preprocess(file_name, new_file_name, encodings):
    for encoding in encodings:
        try:
            with open(file_name, 'r', encoding=encoding) as f1, open(new_file_name, 'w', encoding="utf-8") as f2:
                for line in f1:
                    line = line.replace('"', "")  # Remove double quotes in each line
                    f2.write(line)  # Write the new line to the new file
            return
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Unable to read the file {file_name} with provided encodings.")

def print_usage():
    print(f"Usage: {sys.argv[0]} <input_filename> <output_filename> <duration in seconds>")

def read_csv_with_fallback(filename, encodings):
    for encoding in encodings:
        try:
            return pd.read_csv(filename, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Unable to read the file {filename} with provided encodings.")

def average_data(input_filename, output_filename, duration_user):
    encodings = ["utf-8", "utf-16", "cp1252"]
    df = read_csv_with_fallback(input_filename, encodings)
    df['Timestamp'] = df['Timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').timestamp())  # Convert the timestamp into a number
    df['Timestamp'] = (df['Timestamp'] // duration_user) * duration_user  # Round down to nearest multiple of duration_user
    df_avg = df.groupby('Timestamp', as_index=False).mean()  # Calculate the mean
    df_avg['Timestamp'] = df_avg['Timestamp'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%dT%H:%M:%S'))  # Convert the number back into timestamp
    df_avg.to_csv(output_filename, index=False)  # Save the result to CSV

if __name__ == "__main__":
    input_filename = r"C:\Users\digvi\OneDrive\Desktop\ECEN 403\Export_20220111T000000_20220111T235959.csv"
    output_filename = input_filename + ".out.csv"
    duration_user = 300  # Duration in seconds

    new_input_filename = input_filename + ".new.csv"  # New CSV file after preprocessing

    encodings = ["utf-8", "utf-16", "cp1252"]
    preprocess(input_filename, new_input_filename, encodings)  # Preprocess the input file
    average_data(new_input_filename, output_filename, duration_user)  # Calculate the average data

    print(f"Processed data saved to {output_filename}")
