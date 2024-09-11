import asyncio
import sys
import server

def main():
    local_server = server.Server()
    asyncio.run(local_server.start())

if __name__ == "__main__":
    print("Launching PredictAI Server...")
    main()