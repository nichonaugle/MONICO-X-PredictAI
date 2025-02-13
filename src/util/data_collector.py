from util import get_data_averaging_interval
from pathlib import Path
import asyncio
import aiofiles
import csv

DATA_DIRECTORY = Path(__file__).parent / "data"  # Assuming 'data' is a folder in your project
CSV_STREAMING_DATA = DATA_DIRECTORY / "test_data.csv"

class DataCollector:
    def __init__(self):
        self.current_data_array = []
        self.running_averaged_data_array = []
        self.data_buf = []
        self.buf_head = 0
        self.buf_tail = 0
        self.max_data_buf_size = (60 * get_data_averaging_interval()) - 1  # Proper buffer length
        self.running_time = 1
        self.new_data = False  # Boolean flag instead of int for clarity

    async def start(self) -> None:
        await self.start_data_collection()

    async def start_data_collection(self) -> None:
        """Reads and processes data from a large CSV file asynchronously."""
        async with aiofiles.open(CSV_STREAMING_DATA, mode='r', newline='') as file:
            reader = csv.reader(await file.readlines())  # Efficient batch read
            while (True):
                
                next(reader, None)  # Skip header if CSV has one
                
                for row in reader:
                    if not row:
                        continue  # Skip empty rows
                    
                    timestamp, value = int(row[0]), float(row[1])  # Example parsing

                    # Store data in buffer
                    if len(self.data_buf) >= self.max_data_buf_size:
                        # Buffer is full, compute average and shift data
                        avg_value = sum(self.data_buf) / len(self.data_buf)
                        self.running_averaged_data_array.append(avg_value)

                        # Shift buffer: remove oldest, append newest
                        self.data_buf.pop(0)

                    self.data_buf.append(value)
                    self.buf_tail += 1

                    # Increase running time by 1 units per row
                    self.running_time += 1

                    # Mark new data available
                    self.new_data = True

                # Async sleep to allow other tasks to run
                await asyncio.sleep(1)
                print(self.data_buf)

    async def new_averaged_data_checker(self) -> bool:
        """Checks if there is new averaged data available."""
        return self.new_data

    async def set_new_data_as_read(self, val: bool) -> None:
        """Marks new averaged data as read."""
        self.new_data = val

    async def get_averaged_data_array(self):
        """Returns the running averaged data array."""
        return self.running_averaged_data_array
