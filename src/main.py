import asyncio
import sys
import server
import uvicorn

# Testing Code
# python -m websockets ws://localhost:8000/device/subscribe/ws

async def main():
    # Start the mDNS service and other asynchronous operations
    local_server = server.Server()
    await local_server.start()

    # Run the Uvicorn server asynchronously
    config = uvicorn.Config("api:app", host=local_server.ip_addr, port=int(local_server.port), reload=True)
    uvicorn_server = uvicorn.Server(config)
    await uvicorn_server.serve()
    
if __name__ == "__main__":
    print("Launching PredictAI Server...")
    asyncio.run(main()) 