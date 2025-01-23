import asyncio
import uvicorn
from services import MDNSService
from util import get_server_port, get_ip_addr
import model
from api import run_server
import os

mdns = MDNSService()
model.init()
# Testing Code
# python -m websockets ws://predictai-B5SBS/ws
async def main():
    # Start the mDNS service, load the tensorflow model, and other asynchronous operations 
    try:
        #TODO: add in the loading of the modelfile. If it cannot be done in an async function, put it in the main function below.
        await asyncio.gather(mdns.start(),run_server())
    except KeyboardInterrupt:
        print("Server interupted by user...")
    finally:
        await mdns.stop()

if __name__ == "__main__":
    print("Launching PredictAI Server...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutdown complete!")