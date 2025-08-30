from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from .tools import load_tools

app = FastAPI(title="MCP Tool Server (dummy)")
TOOLS = load_tools()

class InvokeRequest(BaseModel):
    name: str
    payload: Dict[str, Any] | None = None

@app.get("/tools")
async def list_tools():
    return {"tools": list(TOOLS.keys())}

@app.post("/invoke")
async def invoke(req: InvokeRequest):
    fn = TOOLS.get(req.name)
    if not fn:
        return {"error": f"tool '{req.name}' not found"}
    return fn(req.payload) 