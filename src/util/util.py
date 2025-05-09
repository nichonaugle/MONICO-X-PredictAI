import json
import socket
import random
import string
import getmac

def get_ip_addr() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        s.connect(('8.8.8.8', 80))  # Google's public DNS server
        local_ip = s.getsockname()[0]
    except Exception as e:
        print(f"Error getting local IP address: {e}")
        local_ip = None
    finally:
        s.close()
    return local_ip

def get_mac_address() -> str:
    return getmac.get_mac_address()

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

def set_data_streaming(value) -> None:
    with open("./server_config.json", "r+") as f:
        config = json.load(f)
        if config["LIVE_STREAM"] != value:
            config["LIVE_STREAM"] = value
            f.seek(0)
            json.dump(config, f, indent=4)
            f.truncate()
        print(f"Live streaming value set to: {value}")

def get_data_streaming() -> bool:
    with open("./server_config.json", "r") as f:
        config = json.load(f)
        state = config["LIVE_STREAM"]
    return state

def set_prediction_state(value) -> None:
    with open("./server_config.json", "r+") as f:
        config = json.load(f)
        if config["PREDICTION_ACTIVE"] != value:
            config["PREDICTION_ACTIVE"] = value
            f.seek(0)
            json.dump(config, f, indent=4)
            f.truncate()
        print(f"AI Prediction state set to: {value}")

def get_prediction_state() -> bool:
    with open("./server_config.json", "r") as f:
        config = json.load(f)
        state = config["PREDICTION_ACTIVE"]
    return state

def set_data_sending_interval(value) -> None:
    with open("./server_config.json", "r+") as f:
        config = json.load(f)
        if config["SENDING_INTERVAL"] != value:
            config["SENDING_INTERVAL"] = value
            f.seek(0)
            json.dump(config, f, indent=4)
            f.truncate()
        print(f"Sending interval set to: {value} seconds")

def get_data_sending_interval() -> int:
    with open("./server_config.json", "r") as f:
        config = json.load(f)
        state = config["SENDING_INTERVAL"]
    return int(state)

def get_data_averaging_interval() -> int:
    with open("./server_config.json", "r") as f:
        config = json.load(f)
        state = config["AVERAGING_INTERVAL"]
    return int(state)