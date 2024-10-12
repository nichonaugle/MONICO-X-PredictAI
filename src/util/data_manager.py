import pandas

import sys
import datetime
from datetime import date, datetime

filename = "C:/Users/digvi/OneDrive/Desktop/ECEN 403/C4701 1 Day Data.csv" #file name
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

def preprocess(file_name, new_file_name):
    with open(file_name) as f1, open(new_file_name, "w") as f2: 
        for line in f1:
            line = line.strip('"') #takes in the file and removes the double quotes in each line
            f2.write(line) # writes the new file in f2

def print_usage():
    print(f"Usage: {sys.argv[0]} <input_filename> <output_filename> <duration in seconds>")

# duration should be in seconds
def average_data(input_filename, output_filename, duration_user): 
    df = pandas.read_csv(input_filename, encoding="cp1252")
    df2 = df['TimeStamp'].apply(lambda x:datetime.strptime(x, '%m/%d/%Y %H:%M').timestamp()) # changing the timestamp into a number
    df['TimeStamp'] = (df2 // duration_user) * duration_user # retrun an integer instead of floating point ## 19//5 = 3 and 3*5 = 15
    df3 = df.groupby('TimeStamp', as_index = False).mean() # calculating the mean
    df3['TimeStamp'] = df3['TimeStamp'].apply(lambda x:datetime.fromtimestamp(x).strftime('%m/%d/%Y %H:%M')) #converts the number back into timestamp
    df3.to_csv(output_filename)

if __name__ == '__main__': ## if running from command line
    if len(sys.argv) != 4:
        print_usage()
        sys.exit(1)

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    duration_user = int(sys.argv[3]) ## classify as integer
    
    new_input_filename = input_filename + ".new.csv" ## will create a new csv file
    preprocess(input_filename, new_input_filename) 
    average_data(new_input_filename, output_filename, duration_user)
    # average_data(input_filename, output_filename, duration_user)