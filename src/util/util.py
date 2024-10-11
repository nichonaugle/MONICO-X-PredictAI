import json
import socket

def get_ip_addr() -> str:
    # TODO: Check if conencted to wifi in the first place and test out the error ip
    ip_address = socket.gethostbyname(socket.gethostname())
    return ip_address

def get_mdns_device_name() -> str:
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

def get_mdns_service_name() -> str:
    with open("./server_config.json", "r") as f:
        config = json.load(f)
        name = config["MDNS_SERVICE_NAME"]
    return name

def get_mdns_service_type() -> str:
    with open("./server_config.json", "r") as f:
        config = json.load(f)
        name = config["MDNS_SERVICE_TYPE"]
    return name

def get_server_port() -> str:
    with open("./server_config.json", "r") as f:
        config = json.load(f)
        name = config["SERVER_PORT"]
    return name
