from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection
from pipecat.transports.network.webrtc_connection import IceServer
from pipecat.processors.frame_processor import FrameProcessor
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


# --- Pipecat version logging ---
import pipecat
logger.info(f"Pipecat version installed: {pipecat.__version__}")

load_dotenv()

app = FastAPI()
app.mount("/client", SmallWebRTCPrebuiltUI)

pcs_map = {}

class WhisperSTTService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        logger.debug(f"🔑 Whisper API Key set: {'yes' if self.api_key else 'no'}")
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

class WhisperProcessor(FrameProcessor):
    def __init__(self, whisper_service):
        super().__init__()
        self.whisper_service = whisper_service

    async def process_frame(self, frame, direction):
        if frame.audio:
            logger.debug("🎧 Audio frame received, sending to Whisper...")
            transcript = await self.whisper_service.transcribe(frame.audio)
            logger.info(f"📝 Transcription result: {transcript}")
        return frame

async def run_whisper(webrtc_connection: SmallWebRTCConnection):
    logger.info("🎙️ run_whisper() called")

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    logger.debug(f"Transport audio_in_enabled: {transport.params.audio_in_enabled}")
    logger.debug(f"Transport audio_out_enabled: {transport.params.audio_out_enabled}")

    logger.info("🔧 Initializing Whisper service")
    whisper = WhisperSTTService()

    processor = WhisperProcessor(whisper)
    pipeline = Pipeline([
        transport.input(),
        processor,
    ])

    logger.info("🏃 Running pipeline task")
    task = PipelineTask(pipeline)
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/client/")


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    logger.info(f"📨 Received /api/offer: {request}")
    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        conn = pcs_map[pc_id]
        logger.info(f"🔁 Reusing connection: {pc_id}")
        await conn.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        logger.info("🆕 Creating new SmallWebRTCConnection")
        conn = SmallWebRTCConnection(
            ice_servers = [
                IceServer(urls="stun:stun.relay.metered.ca:80"),
                IceServer(urls="turn:global.relay.metered.ca:80", username="aace885d29a1c3dd912192b1", credential="4Uuv6SsCyBMY//QH"),
                IceServer(urls="turn:global.relay.metered.ca:80?transport=tcp", username="aace885d29a1c3dd912192b1", credential="4Uuv6SsCyBMY//QH"),
                IceServer(urls="turn:global.relay.metered.ca:443", username="aace885d29a1c3dd912192b1", credential="4Uuv6SsCyBMY//QH"),
                IceServer(urls="turns:global.relay.metered.ca:443?transport=tcp", username="aace885d29a1c3dd912192b1", credential="4Uuv6SsCyBMY//QH"),
            ]
        )
        logger.info("🧊 Using ICE servers:")
        for s in conn.ice_servers:
            logger.info(f" - {s.urls} (username={s.username})")

        @conn.event_handler("connectionstatechange")
        async def on_state_change(state):
            logger.debug(f"🔄 Connection state update: {state}")

        await conn.initialize(sdp=request["sdp"], type=request["type"])

        # Log each ICE candidate as they are gathered
        @conn.event_handler("icecandidate")
        async def on_ice_candidate(event):
            logger.info(f"🧊 ICE candidate gathered: {event}")

        @conn.event_handler("closed")
        async def handle_close(connection):
            logger.info(f"❌ Closing: {connection.pc_id}")
            pcs_map.pop(connection.pc_id, None)

        logger.info("🚀 Scheduling run_whisper background task")
        background_tasks.add_task(run_whisper, conn)

    answer = conn.get_answer()
    pcs_map[answer["pc_id"]] = conn
    return answer


# Health check endpoint
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


# Test import endpoint for pipecat.pipeline.processor. Must be before return in /api/offer.
@app.get("/test-import")
def test_import():
    from pipecat.processors.frame_processor import FrameProcessor
    return {"status": "import successful"}
    