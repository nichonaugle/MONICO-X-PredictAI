from fastapi import APIRouter
from util import get_server_port, get_mdns_service_name, get_ip_addr, get_mdns_device_name, set_data_streaming, get_data_streaming, set_prediction_state, get_prediction_state, set_data_sending_interval, get_data_sending_interval
from api.ws import get_active_connections
#from model import run_model

router = APIRouter()

@router.get("/server/active-tasks")
def get_device_active_tasks():
    json_response = {"Tasks": "None"}
    return json_response

@router.get("/server/set-streaming")
def get_server_data_streaming(streaming: bool):
    set_data_streaming(streaming)
    json_response = {"Data Live Streaming": f"{get_data_streaming()}"}
    return json_response

@router.get("/server/info")
def get_device_info():
    json_response = {
        "Server Hostname": f"{get_mdns_device_name()}",
        "IP Address": f"{get_ip_addr()}",
        "Server Port": f"{get_server_port()}",
        "mDNS Service": f"{get_mdns_service_name()}",
        "WebSocket Client Count": f"{len(get_active_connections())}",
        "Data Live Streaming": f"{get_data_streaming()}",
        "Data Averaging Interval": f"{get_data_sending_interval()}",
        "AI Prediction State": f"{get_prediction_state()}"
        }
    return json_response

@router.get("/data/set-sending-interval-period")
def get_classification_sending_period(seconds: int):
    set_data_sending_interval(seconds)
    json_response = {"Data Sending Interval": f"{get_data_sending_interval()}"}
    return json_response

@router.get("/ai/set-prediction-state")
def get_ai_prediction_state(state: bool):
    set_prediction_state(state)
    json_response = {"AI Prediction State": f"{get_prediction_state()}"}
    return json_response