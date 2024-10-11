import socket
import random
import string
import json
import asyncio
import websockets
import websockets.asyncio
import websockets.asyncio.server
from zeroconf import IPVersion, ServiceInfo, Zeroconf
from util import get_ip_addr, get_mdns_device_name, get_mdns_service_type, get_mdns_service_name, get_server_port

DEFAULT_SERVICE_TYPE = "_predictai-ws._tcp.local."
DEFAULT_SERVICE_NAME = "Monico PredictAI Hub"
DEFAULT_DEVICE_NAME = "Monico"
DEFAULT_SERVICE_PORT = 80

class MDNSService:
    def __init__(
        self,
        device_name=DEFAULT_DEVICE_NAME,
        service_type=DEFAULT_SERVICE_TYPE,
        service_name=DEFAULT_SERVICE_NAME,
        port=DEFAULT_SERVICE_PORT,
    ):
        self.ip_addr = get_ip_addr()
        self.device_name = get_mdns_device_name()
        self.service_type = get_mdns_service_type()
        self.service_name = get_mdns_service_name()
        self.port = get_server_port()
        self.zc_instance = Zeroconf(ip_version=IPVersion.All)

    async def start(self):
        service_info = ServiceInfo(
            type_=self.service_type,
            name=f"{self.service_name}.{self.service_type}",
            addresses=[socket.inet_aton(str(self.ip_addr))],
            port=int(self.port),
            properties={"API Docs": f"{self.device_name}:{self.port}/docs"},
            server=f"{self.device_name}.local.",
        )
        self.zc_instance.register_service(service_info)
        print(f"Service {self.service_name} started on {self.ip_addr}:{self.port}")
        while True:
            await asyncio.Future()

    async def stop(self):
        # Unregister the service when done
        self.zc_instance.unregister_all_services()
        self.zc_instance.close()
        print(f"Service {self.service_name} stopped.")
