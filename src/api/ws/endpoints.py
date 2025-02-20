from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.requests import Request
from .connection_manager import ConnectionManager
from model import run_model
from asyncio import sleep, create_task
import random
from util import DataCollector
import aiofiles
import json

router = APIRouter()
connection_manager = ConnectionManager()

def get_active_connections() -> list[WebSocket]:
    return connection_manager.active_connections

async def broadcast_data(websocket):
    data_collector = websocket.app.state.data_collector
    prediction = "n/a"
    while connection_manager.active_connections:  # Only run if clients are connected
        async with aiofiles.open("./server_config.json", "r") as f:
            content = await f.read()
            config = json.loads(content)
            delay = config["SENDING_INTERVAL"]
            stream_state = config["LIVE_STREAM"]
            prediction_state = config["PREDICTION_ACTIVE"]
        if stream_state:
            if prediction_state and await data_collector.new_averaged_data_checker():
                prediction = run_model(data_collector.running_averaged_data_array)
                data_collector.set_new_data_as_read()

            message = f"Collected Data: {data_collector.current_data_array}, Predicted Failure: {prediction} hrs"
            await connection_manager.broadcast(message)

        await sleep(delay)

@router.websocket("/ws")
async def websocket_route(websocket: WebSocket):
    await connection_manager.client_connect(websocket)
    print("Client Connected")

    # Start broadcasting only if it's the first client
    if len(connection_manager.active_connections) == 1:
        create_task(broadcast_data(websocket))

    try:
        await websocket.receive_text()  # Keep the connection alive
    except WebSocketDisconnect:
        connection_manager.client_disconnect(websocket)
        await connection_manager.broadcast("Client has left the chat")
