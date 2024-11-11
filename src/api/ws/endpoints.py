from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from .connection_manager import ConnectionManager

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
            data = await websocket.receive_text()
            await connection_manager.send_ws_client_data(f"You wrote: {data}", websocket)
            await connection_manager.broadcast(f"Client says: {data}")
    except WebSocketDisconnect:
        connection_manager.client_disconnect(websocket)
        await connection_manager.broadcast(f"Client has left the chat")