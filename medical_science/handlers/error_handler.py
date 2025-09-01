from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(413)
    async def request_entity_too_large(request: Request, exc):
        return JSONResponse(status_code=413, content={"error": "Payload too large"}) 