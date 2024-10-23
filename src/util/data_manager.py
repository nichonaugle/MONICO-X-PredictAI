import pandas as pd
from datetime import datetime

def preprocess(file_name, new_file_name, encodings):
    for encoding in encodings:
        try:
            with open(file_name, 'r', encoding=encoding) as f1, open(new_file_name, 'w', encoding="utf-8") as f2:
                for line in f1:
                    line = line.replace('"', "")  # removing the double quotes in each line
                    f2.write(line)  # writing the new line to the new file
            return
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Unable to read the file {file_name} with provided encodings.") #just in case the encoding does not work

    
#function to check through different encodings as each file has different mainly utf-8
def read_csv_with_fallback(filename, encodings):
    for encoding in encodings:
        try:
            return pd.read_csv(filename, encoding=encoding) #going to try the encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Unable to read the file {filename} with provided encodings.")

    
#going to store the non-numerical values in different columns with the assigned names

def find_and_store_non_numerical(df):
    # dictionary to store non-numeric values for each column
    non_numeric_data = {col: ['' for _ in range(len(df))] for col in df.columns[1:]}

    # iterating over the DataFrame columns, starting from the second column (assuming 'Timestamp' is first)
    for column in df.columns[1:]:
        # converting each column to numeric, forcing errors to NaN
        non_numeric = pd.to_numeric(df[column], errors='coerce')
        
        #NaN is going to fill the non-numerical values so that it does not count while averaging the data
        
        # getting the indices where non-numeric values were found
        non_numeric_indices = non_numeric[non_numeric.isna()].index
        
        # for each index with non-numeric values, append the value to the corresponding position in non_numeric_data
        for idx in non_numeric_indices:
            non_numeric_data[column][idx] = df.loc[idx, column]
        
        # replacing the non-numeric values in the original column with NaN
        df[column] = non_numeric

    # creating new columns in the DataFrame for each column that had non-numeric values
    for column, values in non_numeric_data.items():
        #adding the column only if non-numerical values are present
        if any(values):  
            df[f'Non_Numeric_Values_{column}'] = values

    return df

#code for a different timestamp
# df = pandas.read_csv(filename)  #reading the source data file
# df2 = df.groupby('TimeStamp').mean() #calculating the mean of the data depending on the Timestamp
# df2.to_csv("C:/Users/digvi/OneDrive/Desktop/ECEN 403/C4701 1 Day Data Output.csv") #exporting the new file into a csv file

# print(df2.head())

# df2 = df['TimeStamp']
# df3 = df2.apply(lambda x:datetime.strptime(x, '%m/%d/%Y %H:%M').timestamp()) ## changing the timestamp into a number
# df2.head(70)
# df2.apply(date.fromtimestamp)

# df['TimeStamp'] = df3//300
# df.head(10)
# df.groupby('TimeStamp').mean()

def detect_fault_relay(df):
    # checking if the column 'Ramsey C4701E.Fault Relay' exists in the DataFrame
    if 'Ramsey C4701E.Fault Relay' in df.columns:
        # finding rows where the Fault Relay is 1 and print the corresponding Timestamp
        fault_times = df[df['Ramsey C4701E.Fault Relay'] == 1]['Timestamp']
        if not fault_times.empty:
            print(f"Fault detected at the following times:\n{fault_times.to_list()}")
        else:
            print("No faults detected.")
    else:
        print("'Ramsey C4701E.Fault Relay' column not found in the data.")

def average_data(input_filename, output_filename, duration_user):
    encodings = ["utf-8", "utf-16", "cp1252"]
    df = read_csv_with_fallback(input_filename, encodings)
    
    # finding non-numeric values and store them in new columns
    df = find_and_store_non_numerical(df)
    
    # processing Timestamp and calculate averages
    df['Timestamp'] = df['Timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').timestamp())  # Convert the timestamp into a number
    df['Timestamp'] = (df['Timestamp'] // duration_user) * duration_user  # Round down to nearest multiple of duration_user
    df_avg = df.groupby('Timestamp', as_index=False).mean()  # Calculate the mean for numeric columns
    
    # converting the timestamp back to a readable format
    df_avg['Timestamp'] = df_avg['Timestamp'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%dT%H:%M:%S'))
    
    # adding the Non_Numeric_Values from the original DataFrame to the averaged DataFrame
    for column in df.columns:
        if column.startswith("Non_Numeric_Values_"):
            # using `first()` here assumes that all non-numeric values in each group are the same
            df_avg[column] = df.groupby('Timestamp')[column].first().reset_index(drop=True)
    
    # saving the averaged data along with non-numeric values
    df_avg.to_csv(output_filename, index=False)

if __name__ == "__main__":
    input_filename = r"C:\Users\digvi\OneDrive\Desktop\ECEN 403\Export_20220111T000000_20220111T235959.csv"
    output_filename = input_filename + ".out.csv"
    duration_user = 300  # duration in seconds can be changed

    # new CSV file after preprocessing with no "" and will be in column format instead of 1 row format as the original file
    
    new_input_filename = input_filename + ".new.csv"  

    encodings = ["utf-8", "utf-16", "cp1252"] #all the encodings which will be checked
    preprocess(input_filename, new_input_filename, encodings)  # preprocessing the input file
    average_data(new_input_filename, output_filename, duration_user)  # calculating the average data

    print(f"Processed data saved to {output_filename}")
