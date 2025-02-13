from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from .connection_manager import ConnectionManager
from model import run_model
from asyncio import sleep
import random
import aiofiles
import json

router = APIRouter()
connection_manager = ConnectionManager()

def get_active_connections() -> list[WebSocket]:
    return connection_manager.active_connections

@router.websocket("/ws")
async def websocket_route(websocket: WebSocket):
    await connection_manager.client_connect(websocket)
    print("Connected")
    try:
        prediction = "n/a"
        async with aiofiles.open("./server_config.json", "r") as f:
                content = await f.read()
                config  = json.loads(content)
                delay = config["SENDING_INTERVAL"]
                stream_state = config["LIVE_STREAM"]
                prediction_state = config["PREDICTION_ACTIVE"]
        while True:
            if stream_state:
                if data_collector.new_data and prediction_state:
                    prediction = run_model(data_collector.running_averaged_data_array)
                    data_collector.set_new_data_as_read()
                message = f"Collected Data: test, Predicted Failure: {prediction} hrs"
                await connection_manager.broadcast(message)
            await sleep(delay)
    except WebSocketDisconnect:
        connection_manager.client_disconnect(websocket)
        await connection_manager.broadcast(f"Client has left the chat")