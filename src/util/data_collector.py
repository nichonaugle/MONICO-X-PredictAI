from util import get_data_averaging_interval

class DataCollector():
    def __init__():
        current_data_array = []
        running_averaged_data_array = []
        data_buf = []
        buf_head = 0
        buf_tail = 0
        max_data_buf_size = (60 * get_data_averaging_interval()) - 1 # Proper buffer length
        running_time = 0
        new_data = 0

    async def start() -> None:
        self.start_data_collection()

    async def start_data_collection() -> None:
        # open massive 350MB csv file
        while(1):
            # get the data from the csv using running timestamp

            # if buffer is full 
                # average data and store in running array
                # Shift data into the buffer
                # move head and tail
            
            # else
                # Shift data into the buffer
                # move tail
            
            # increase running time variable by ten so it only gets every tenth index
    
    async def new_averaged_data_checker() -> bool:
        return self.new_data
    
    async def set_new_data_as_read(val) -> None:
        self.new_data = val

    async def get_averaged_data_array():
        return self.running_averaged_data_array