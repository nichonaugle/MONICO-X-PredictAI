from fastapi import APIRouter
from util import get_server_port, get_mdns_service_name, get_ip_addr
#from model import run_model

router = APIRouter()

@router.get("/device/active-tasks")
def get_device_active_tasks():
    json_response = {"Tasks": "None"}
    return json_response

@router.get("/device/info")
def get_device_info():
    clients_info = []

    json_response = {
        "Name": "Test",
        "IP Address": f"{get_ip_addr()}",
        "Server Port": f"{get_server_port()}",
        "mDNS Service": f"{get_mdns_service_name()}",
        "WebSocket Clients": "Not Avaliable"
        }
    return json_response

@router.get("/classification/averaging-interval-period")
def get_classification_averaging_period():
    json_response = {"Data Averaging Interval": "5m"}
    return json_response

@router.get("/ai/retrain_model")
def get_classification_averaging_period():
    json_response = {"Data Averaging Interval": "5m"}
    return json_response

@router.get("/ai/prediction_run_state")
def get_classification_averaging_period():
    json_response = {"Data Averaging Interval": "5m"}
    return json_response

@router.get("/ai/prediction_run/{interval}/{timeout}")
def get_classification_averaging_period(interval: str, timeout: int):
    json_response = {
        "Data Prediction Interval Received": interval,
        "Data Prediction Timeout Received": timeout
    }
    return run_model()