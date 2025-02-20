import util
from pathlib import Path
import asyncio
import aiofiles
import csv
import io

UTIL_DIRECTORY = Path(__file__).parent  # Assuming 'data' is a folder in your project
SRC_DIRECTORY = Path(UTIL_DIRECTORY).parent
ROOT_DIRECTORY = Path(SRC_DIRECTORY).parent
CSV_STREAMING_DATA = ROOT_DIRECTORY / "data/STREAMING_DATA_SIM.csv"

class DataCollector:
    def __init__(self):
        self.current_data_array = []
        self.running_averaged_data_array = []
        self.data_buf = []
        self.buf_cycles = 0
        self.max_data_buf_size = (60 * util.get_data_averaging_interval())  # Proper buffer length
        self.running_time = 1
        self.new_data = False  # Boolean flag instead of int for clarity

    async def start(self) -> None:
        await self.start_data_collection_simulator()

    async def start_data_collection_simulator(self) -> None:
        """Reads and processes data from a large CSV file asynchronously."""
        print("Opening File...")
        async with aiofiles.open(CSV_STREAMING_DATA, mode='r', newline='', encoding='utf-8', errors="replace") as f:
            raw_data = await f.read()  # Read entire file asynchronously
            cleaned_data = raw_data.replace('\x00', '')  # Remove null bytes
            reader = csv.reader(io.StringIO(cleaned_data))  # Efficient batch read
            print("Internal Data Streaming Simualtor Started")
            while (True):
                next(reader, None)  # Skip header if CSV has one
                for row in reader:
                    if not row:
                        continue  # Skip empty rows
                    if self.data_buf == []:
                        for idx in range(0, len(row)):
                            self.data_buf.append(0)
                            self.running_averaged_data_array.append(0)
                    else:
                        self.current_data_array = row
                        if self.buf_cycles == self.max_data_buf_size:
                            for i in range(0, len(self.data_buf)):
                                self.running_averaged_data_array[i] = self.data_buf[i] / self.max_data_buf_size
                                self.data_buf[i] = 0 
                            self.buf_cycles = 0
                            self.new_data = True
                        for idx in range(1, len(row)):
                            try:
                                val = float(row[idx])
                            except:
                                val = 0
                            self.data_buf[idx] = self.data_buf[idx] + val
                        self.buf_cycles += 1
                    await asyncio.sleep(1)

    async def new_averaged_data_checker(self) -> bool:
        """Checks if there is new averaged data available."""
        return self.new_data

    async def set_new_data_as_read(self) -> None:
        """Marks new averaged data as read."""
        self.new_data = False

    async def get_averaged_data_array(self):
        """Returns the running averaged data array."""
        return self.running_averaged_data_array
    
    async def get_current_data_array(self):
        """Returns the running averaged data array."""
        return self.current_data_array
