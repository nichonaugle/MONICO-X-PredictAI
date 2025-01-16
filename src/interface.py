import asyncio
import websockets
import socket
from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
from util import get_mdns_service_type

class MyListener(ServiceListener):

    def __init__(self):
        self.services = {}

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        print(f"Service {name} updated")

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        print(f"Service {name} removed")

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        if info:
            ip = socket.inet_ntoa(info.addresses[0])
            server = info.server
            self.services[name] = {'ip': ip, 'server': server}
            print(f"{name.split('.')[0]} found -> IP: {ip}  Server Name: {server.split('.')[0]}")

async def connect_to_service(hostname: str):  # TODO: Doesnt work
    uri = f"ws://{hostname}/ws"
    async with websockets.connect(uri) as websocket:
        print(f"Connected to WebSocket at {uri}")
        
        while True:
            response = await websocket.recv()
            print(f"Received: {response}")
        
def main():
    zeroconf = Zeroconf()
    listener = MyListener()
    browser = ServiceBrowser(zeroconf, f"{get_mdns_service_type()}", listener)

    try:
        print("Scanning for services...")
        input("Press Enter to stop scanning...\n")

        if listener.services:
            print("\nAvailable services:")

            for idx, (name, ip) in enumerate(listener.services.items(), 1):
                print(f"{idx}. {name} - {ip}")

            while True:
                try:
                    selection = int(input(f"Select a service to connect (1-{len(listener.services)}): "))
                    if 1 <= selection <= len(listener.services):
                        break  # Valid selection
                    else:
                        print(f"Please enter a number between 1 and {len(listener.services)}.")
                except ValueError:
                    print("Invalid input. Please enter a valid number.")
            
            selected_service = list(listener.services.items())[selection - 1]
            service_name, service_info = selected_service
            service_ip = service_info['ip']
            service_server = service_info['server']

            print(f"Connecting to {service_name} (Server: {service_server.split('.')[0]}) at {service_ip}...")

            asyncio.run(connect_to_service(service_server.split(".")[0]))
        else:
            print("No services found.")
    except KeyboardInterrupt:
        print("Server interupted by user, shutting down...")

    finally:
        zeroconf.close()

if __name__ == "__main__":
    main()