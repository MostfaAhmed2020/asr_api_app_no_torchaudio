from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import pipeline
import uvicorn
import tempfile
import subprocess
import numpy as np
from scipy.io import wavfile

app = FastAPI()

# Load ASR model once on startup
asr = pipeline("automatic-speech-recognition", model="distil-whisper/distil-small.en")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp_input:
            contents = await file.read()
            tmp_input.write(contents)
            tmp_input_path = tmp_input.name

        # Convert audio to .wav with ffmpeg
        tmp_output_path = tmp_input_path + ".wav"
        command = [
            "ffmpeg", "-y", "-i", tmp_input_path,
            "-ar", "16000",  # 16kHz for Whisper
            "-ac", "1",      # mono
            "-f", "wav",
            tmp_output_path
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # Read WAV file
        sampling_rate, audio_data = wavfile.read(tmp_output_path)

        # Normalize audio if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max

        # Transcribe
        result = asr(audio_data, sampling_rate=sampling_rate)
        return JSONResponse(content={"text": result["text"]})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
