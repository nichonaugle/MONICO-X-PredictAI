"""
FOR AI
'train_model'
'prediction_run_state'
'prediction_run(interval, timeout)'

FOR CLASSIFICATION
'averaging_interval_peiod'

FOR API
'device_info'
'active_tasks'
'subscribe(interrval, timeout)'
'unsubscibe'


"""
import asyncio
from typing import Union
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def client_connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def client_disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_ws_client_data(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

app = FastAPI()
manager = ConnectionManager()

@app.websocket("/ws")
async def subscribe_data_stream(websocket: WebSocket):
    await manager.client_connect(websocket)
    print("Connected")
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_ws_client_data(f"You wrote: {data}", websocket)
            await manager.broadcast(f"Client says: {data}")
    except WebSocketDisconnect:
        manager.client_disconnect(websocket)
        await manager.broadcast(f"Client has left the chat")

@app.get("/device/active-tasks")
def get_device_active_tasks():
    json_response = {"Tasks": "None"}
    return json_response

@app.get("/device/info")
def get_device_info():
    json_response = {"Name": "Test"}
    return json_response

@app.get("/classification/averaging-interval-period")
def get_classification_averaging_period():
    json_response = {"Data Averaging Interval": "5m"}
    return json_response
