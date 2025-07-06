from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection
from pipecat.transports.network.webrtc_connection import IceServer
from pipecat.pipeline.processor import Processor
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.base_transport import TransportParams
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI
from loguru import logger
from typing import Optional
import aiohttp
import os

load_dotenv()

app = FastAPI()
app.mount("/client", SmallWebRTCPrebuiltUI)

pcs_map = {}
ice_servers = [IceServer(urls="stun:stun.l.google.com:19302")]


class WhisperSTTService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("Missing OPENAI_API_KEY")
            raise ValueError("Missing OPENAI_API_KEY")

    async def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = aiohttp.FormData()
        data.add_field("file", audio_bytes, filename=filename, content_type="audio/wav")
        data.add_field("model", "whisper-1")

        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.openai.com/v1/audio/transcriptions", headers=headers, data=data) as resp:
                resp.raise_for_status()
                result = await resp.json()
                return result["text"]

class WhisperProcessor(Processor):
    def __init__(self, whisper_service):
        self.whisper_service = whisper_service

    async def process(self, frame):
        if frame.audio:
            transcript = await self.whisper_service.transcribe(frame.audio)
            print(f"User said: {transcript}")
            logger.info(f"User said: {transcript}")
        return frame

async def run_whisper(webrtc_connection: SmallWebRTCConnection):
    logger.info("Bot started (whisper-only)")

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    whisper = WhisperSTTService()

    processor = WhisperProcessor(whisper)
    pipeline = Pipeline([
        transport.input(),
        processor,
    ])

    task = PipelineTask(pipeline)
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/client/")


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        conn = pcs_map[pc_id]
        logger.info(f"Reusing connection: {pc_id}")
        await conn.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        conn = SmallWebRTCConnection(ice_servers)
        await conn.initialize(sdp=request["sdp"], type=request["type"])

        @conn.event_handler("closed")
        async def handle_close(connection):
            logger.info(f"Closing: {connection.pc_id}")
            pcs_map.pop(connection.pc_id, None)

        background_tasks.add_task(run_whisper, conn)

    answer = conn.get_answer()
    pcs_map[answer["pc_id"]] = conn
    return answer


# Health check endpoint
@app.get("/healthz")
def healthz():
    return {"status": "ok"}