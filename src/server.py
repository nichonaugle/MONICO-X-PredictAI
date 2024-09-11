import socket
import random
import string
import json
import asyncio
import websockets
from zeroconf import IPVersion, ServiceInfo, Zeroconf
from flask_sockets import Sockets
from flask import Flask

app = Flask(__name__)
sockets = Sockets(app)
zeroconf = Zeroconf(ip_version=IPVersion.All)

class Server():
    def __init__(self):
        self.ip_addr = self.get_ip_addr()
        self.clients = {}
        self.port = self.get_websocket_server_port()
        self.mdns_device_name = self.get_mdns_device_name()
        self.mdns_service = self.get_mdns_service_name()
        self.mdns_port = "5353"

    async def start(self):
        print("Starting server (asynchronously)")
        await asyncio.gather(
            self.launch_mdns(),
            self.launch_socket_server()
        )

    async def launch_mdns(self):
        print("Launching mDNS service")
        my_service = ServiceInfo(
            self.mdns_service,
            (self.mdns_device_name + "." + self.mdns_service),
            addresses=[socket.inet_aton(self.ip_addr)],
            port=int(self.port),
        )
        zeroconf.register_service(my_service)
        print(f"Successfully launched mDNS as: '{self.mdns_service}'")
        while True:
            await asyncio.Future()

    async def launch_socket_server(self):
        server = await websockets.serve(self.client_handler, self.ip_addr, int(self.port))
        print(f"WebSocket server started at ws://{self.ip_addr}:{self.port}")
        await server.wait_closed()

    async def start_sending_data_to_client(self, ws, client_id):
        while not ws.closed and self.clients[client_id]["state"] == "start":
            ### SEND TEST DATA ###
            sensor_data = {
                'temperature': random.uniform(20.0, 30.0),
                'humidity': random.uniform(30.0, 50.0)
            }
            await ws.send(json.dumps(sensor_data))
            await asyncio.sleep(1)

    async def client_handler(self, ws):
        print("New client connected")
        client_id = id(ws)
        self.clients[client_id] = {"client-id": client_id, "state": "stop"}
        try:
            while not ws.closed:
                message = await ws.receive()
                if message == "start":
                    self.clients[client_id]["state"] = "start"
                    await self.start_sending_data_to_client(ws, client_id)
                elif message == "stop":
                    self.clients[client_id]["state"] = "stop"
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
        finally:
            del self.clients[client_id]

    def get_ip_addr(self) -> str:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        return local_ip

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