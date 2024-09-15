import socket
import random
import string
import json
import asyncio
import websockets
import websockets.asyncio
import websockets.asyncio.server
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
            self.mdns_service,#"_http._tcp.local.",#self.mdns_service,
            (self.mdns_device_name + "." + self.mdns_service),#"balls._http._tcp.local.",#(self.mdns_device_name + "." + self.mdns_service),
            addresses=[socket.inet_aton("127.0.0.1")],
            port=int(self.port),
        )
        zeroconf.register_service(my_service)
        print(f"Successfully launched mDNS as: '{self.mdns_service}'")
        while True:
            await asyncio.Future()

    async def launch_socket_server(self):
        async with websockets.asyncio.server.serve(self.client_handler, self.ip_addr, int(self.port)):
            print(f"WebSocket server started at ws://{self.ip_addr}:{self.port}")
            await asyncio.get_running_loop().create_future()

    async def start_sending_data_to_client(self, ws, client_id):
        while ws and self.clients[client_id]["state"] == "start":
            ### SEND TEST DATA ###
            sensor_data = {
                'temperature': random.uniform(20.0, 30.0),
                'humidity': random.uniform(30.0, 50.0)
            }
            await ws.send(json.dumps(sensor_data))
            await asyncio.sleep(1)

    # run "python -m websockets ws://192.168.56.1:8024" to test a client interface
    #TODO Add API callbacks for recieving serial data to handle multiple commands
    async def client_handler(self, ws):
        print("New client connected")
        client_id = id(ws)
        self.clients[client_id] = {"client-id": client_id, "state": "stop", "sending_task": None}
        try:
            async for message in ws:
                if message == "start":
                    self.clients[client_id]["state"] = "start"
                    self.clients[client_id]["sending_task"] = asyncio.create_task(self.start_sending_data_to_client(ws, client_id))
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