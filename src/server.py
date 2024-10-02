import socket
import random
import string
import json
import asyncio
import websockets
import websockets.asyncio
import websockets.asyncio.server
from zeroconf import IPVersion, ServiceInfo, Zeroconf

zeroconf = Zeroconf(ip_version=IPVersion.All)

class DeviceServer():
    def __init__(self):
        self.ip_addr = self.get_ip_addr()
        self.port = self.get_websocket_server_port()
        self.mdns_device_name = self.get_mdns_device_name()
        self.mdns_service = self.get_mdns_service_name()

    # Tested
    async def start(self):
        print("Starting server (asynchronously)")
        asyncio.create_task(self.launch_mdns())

    # Tested
    async def launch_mdns(self):
        print("Launching mDNS service")
        desc = {"API": f"{self.ip_addr}:{self.port}/docs"}
        my_service = ServiceInfo(
            self.mdns_service,
            (self.mdns_device_name + "." + self.mdns_service),
            addresses=[socket.inet_aton(str(self.ip_addr))],
            port=int(self.port),
            properties=desc,
            server=f"{self.mdns_device_name}.local."
        )
        zeroconf.register_service(my_service)
        print(f"Successfully launched mDNS as: '{self.mdns_service}'")
        while True:
            await asyncio.Future()

    def get_ip_addr(self) -> str:
        try:
            # Create a socket connection to an external address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Connect to an external address (this doesn't actually send packets)
            s.connect(("8.8.8.8", 80))  # Google's public DNS
            ip_address = s.getsockname()[0]
            s.close()
            return ip_address
        except Exception as e:
            return str(e)

    def get_mdns_device_name(self) -> str:
        with open("./server_config.json", "r+") as f:
            config = json.load(f)
            if config["MDNS_DEVICE_NAME"] == "predictai-name-not-generated":
                name = "predictai-" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
                print(f"No mdns service name found. New name created as: '{name}'")
                config["MDNS_DEVICE_NAME"] = name
                f.seek(0)
                json.dump(config, f, indent=4)
                f.truncate()
                print("Mdns service name added to server_config.json")
            else:
                name = config["MDNS_DEVICE_NAME"]
                print(f"Set mdns service name from server_config.json as: '{name}'")
        return name

    def get_mdns_service_name(self) -> str:
        with open("./server_config.json", "r") as f:
            config = json.load(f)
            name = config["MDNS_SERVICE_NAME"]
        return name

    def get_websocket_server_port(self) -> str:
        with open("./server_config.json", "r") as f:
            config = json.load(f)
            name = config["WEBSOCKET_SERVER_PORT"]
        return name