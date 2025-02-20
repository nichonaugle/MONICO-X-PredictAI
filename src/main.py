import asyncio
import uvicorn
from services import MDNSService
from util import get_server_port, get_ip_addr, DataCollector
import model
from api import run_server
import os

if __name__ == "__main__":
    print("Launching PredictAI Server...")
    model.init()
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("Shutdown complete!")