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
from fastapi import FastAPI
from api.http import router as http_router
from api.ws import router as ws_router
import uvicorn
import asyncio
from util import get_server_port, get_ip_addr

app = FastAPI(title="Monico API", version="0.0.1")
app.include_router(http_router, prefix="", tags=["http"])
app.include_router(ws_router, prefix="", tags=["websockets"])

async def run_server():
    config = uvicorn.Config(app, host=str(get_ip_addr()), port=int(get_server_port()), reload=True)
    uvicorn_server = uvicorn.Server(config)
    await uvicorn_server.serve()