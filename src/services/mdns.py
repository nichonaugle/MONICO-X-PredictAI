import socket
from zeroconf.asyncio import AsyncZeroconf, AsyncServiceInfo
from zeroconf import IPVersion
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
        self.service_info = AsyncServiceInfo(
            type_=self.service_type,
            name=f"{self.service_name}.{self.service_type}",
            addresses=[socket.inet_aton(str(self.ip_addr))],
            port=int(self.port),
            properties={"API Docs": f"{self.device_name}:{self.port}/docs"},
            server=f"{self.device_name}.local.",
        )
        self.zc_instance = None
        self.zc_task = None

    async def start(self):
        self.zc_instance = AsyncZeroconf(ip_version=IPVersion.All)
        self.zc_task = await self.zc_instance.async_register_service(self.service_info, allow_name_change=True)
        print(f"Service {self.service_name} started on {self.ip_addr}:{self.port}")

    async def stop(self):
        # Unregister the service when done
        await self.zc_instance.async_unregister_all_services()
        await self.zc_instance.async_close()
        print(f"Service {self.service_name} stopped.")
