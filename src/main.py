import asyncio
import uvicorn
from services import MDNSService
from util import get_server_port, get_ip_addr
from api import run_server

mdns = MDNSService()
# Testing Code
# python -m websockets ws://predictai-B5SBS/ws
async def main():
    # Start the mDNS service and other asynchronous operations 

    await asyncio.gather(mdns.start(),run_server())
    
if __name__ == "__main__":
    print("Launching PredictAI Server...")
    asyncio.run(main())