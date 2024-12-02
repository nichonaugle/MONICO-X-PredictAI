from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from .connection_manager import ConnectionManager
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
        while True:
            async with aiofiles.open("./server_config.json", "r") as f:
                content = await f.read()
                config  = json.loads(content)
                delay = config["AVERAGING_INTERVAL"]
                stream_state = config["LIVE_STREAM"]
                prediction_state = config["PREDICTION_ACTIVE"]
            await sleep(delay)
            if stream_state:
                collected_data = random.randint(1, 100)
                prediction = random.randint(1, 10) if prediction_state else "N/A"
                message = f"Collected Data: {collected_data}, Predicted Failure: {prediction} hrs"
                await connection_manager.broadcast(message)
    except WebSocketDisconnect:
        connection_manager.client_disconnect(websocket)
        await connection_manager.broadcast(f"Client has left the chat")