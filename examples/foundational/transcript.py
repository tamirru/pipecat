from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.base_transport import TransportParams
from pipecat.services.openai.whisper import WhisperSTTService
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI
from loguru import logger
import os

load_dotenv()

app = FastAPI()
app.mount("/client", SmallWebRTCPrebuiltUI)

pcs_map = {}
ice_servers = [IceServer(urls="stun:stun.l.google.com:19302")]

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

    stt = WhisperSTTService()

    async def print_transcript(frame):
        if frame.text:
            print(f"User said: {frame.text}")

    stt.event_handler("on_text")(print_transcript)

    pipeline = Pipeline([
        transport.input(),
        stt
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