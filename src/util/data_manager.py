import pandas
import argparse
import sys
import datetime
from datetime import date, datetime
import os

cwd = os.getcwd()
DEFAULT_FILENAME = f"{cwd}/data/test_data.csv"
'''
# Get parsed passed in arguments.
parser = argparse.ArgumentParser(
    description="Data Manager"
)

parser.add_argument(
    "--input-file-path",
    type=str,
    default=DEFAULT_FILENAME,
    help=f"Input fiolename to parse. Defaults to: {DEFAULT_FILENAME}",
)

args = parser.parse_args()
'''

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

def preprocess(file_name, test_filename):
    with open(file_name) as f1, open(test_filename, "w") as f2: 
        for line in f1:
            line = line.replace('"', "") #takes in the file and removes the double quotes in each line
            f2.write(line) # writes the new file in f2

def print_usage():
    print(f"Usage: {sys.argv[0]} <input_filename> <output_filename> <duration in seconds>")

# duration should be in seconds
def average_data(input_filename, output_filename, duration_user):
    df = pandas.read_csv(input_filename, encoding="cp1252")
    df2 = df['TimeStamp'].apply(lambda x:datetime.strptime(x, '%Y/%m/%d %H:%M').timestamp()) # changing the timestamp into a number
    df['TimeStamp'] = (df2 // duration_user) * duration_user # retrun an integer instead of floating point ## 19//5 = 3 and 3*5 = 15
    print(df['TimeStamp'])
    df3 = df.groupby('TimeStamp', as_index = False).mean() # calculating the mean
    df3['TimeStamp'] = df3['TimeStamp'].apply(lambda x:datetime.fromtimestamp(x).strftime('%m/%d/%Y %H:%M')) #converts the number back into timestamp
    df3.to_csv(output_filename)

if __name__ == '__main__': ## if running from command line
    if len(sys.argv) != 4:
        print_usage()
        sys.exit(1)

    input_filename = sys.argv[1]
    print(input_filename)
    output_filename = sys.argv[2]
    print(output_filename)
    duration_user = int(sys.argv[3]) ## classify as integer
    print(duration_user)

    test_filename = "test.csv" ## will create a new csv file
    preprocess(input_filename, test_filename) 
    timestamp_test = "2022-01-01T00:00:00"
    print(datetime.strptime(timestamp_test, '%Y/%m/%d %H:%M').timestamp())
    #average_data(test_filename, output_filename, duration_user)
    # average_data(input_filename, output_filename, duration_user)