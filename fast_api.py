from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi import BackgroundTasks
import numpy as np
import soundfile as sf
import tempfile
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Keep your original imports here
import io
import re
import tempfile
from pathlib import Path
from typing import Optional
import os
from pathlib import Path
from cached_path import cached_path

import numpy as np
import soundfile as sf

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT, UNetT

# Configuration 
MODEL_NAME = "F5-TTS-small"
REF_AUDIO = "/home/cmi_10101/Documents/voice/F5_TTS/src/f5_tts/ref_audio_text_files/ref_audio_1.wav"
REF_TEXT = '''हाँ भैया अभी दसेहरी आम में दसेहरी रखें हैं  ओर केले भी हैं हमारे पास अभी मतलब बताइये आपको कित्ते के चाहिए|'''
CKPT_FILE = "/home/cmi_10101/Documents/voice/F5_TTS/ckpts/model_10000.pt"
VOCAB_FILE = "/home/cmi_10101/Documents/voice/F5_TTS/data/vocab.txt"
VOCOS_LOCAL_PATH = "../ckpts/vocos-mel-24khz"

# Model parameters
MODEL_CLS = DiT
MODEL_CFG = dict(dim=768, depth=18, heads=12, ff_mult=2, text_dim=512, conv_layers=4)
MEL_SPEC_TYPE = "vocos"
REMOVE_SILENCE = True
SPEED = 1.0
NFE_STEP = 32
INDIC = True

# Global variables to hold loaded models
loaded_models = {}

class TTSRequest(BaseModel):
    text: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models during startup
    print("Loading TTS model...")
    try:
        # Load vocoder
        vocoder = load_vocoder(
            vocoder_name=MEL_SPEC_TYPE,
            is_local=False,
            local_path=VOCOS_LOCAL_PATH
        )
        
        # Load TTS model
        ema_model = load_model(
            MODEL_CLS, 
            MODEL_CFG, 
            CKPT_FILE, 
            mel_spec_type=MEL_SPEC_TYPE, 
            vocab_file=VOCAB_FILE
        )
        
        # Preprocess reference audio and text
        processed_ref_audio, processed_ref_text = preprocess_ref_audio_text(
            REF_AUDIO, REF_TEXT
        )
        
        loaded_models.update({
            "model": ema_model,
            "vocoder": vocoder,
            "processed_ref_audio": processed_ref_audio,
            "processed_ref_text": processed_ref_text
        })
        
        print("Models loaded successfully")
        yield
    
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise e

app = FastAPI(lifespan=lifespan)

def generate_audio(text: str) -> str:
    try:
        temp_dir = tempfile.mkdtemp()
        output_file = Path(temp_dir) / "generated_audio.wav"
        
        # Modified infer_process call 
        audio, final_sample_rate, _ = infer_process(
            loaded_models["processed_ref_audio"],
            loaded_models["processed_ref_text"],
            text,
            loaded_models["model"],
            loaded_models["vocoder"],
            mel_spec_type=MEL_SPEC_TYPE,
            speed=SPEED,
            nfe_step=NFE_STEP
        )  
        
        sf.write(str(output_file), audio, final_sample_rate)
        
        if REMOVE_SILENCE:
            remove_silence_for_generated_wav(str(output_file))
            
        return str(output_file)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio generation failed: {str(e)}"
        )

@app.post("/generate-tts/")
async def generate_tts(
    request: TTSRequest,
    background_tasks: BackgroundTasks
):
    output_path = None
    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input text cannot be empty"
            )
        
        # Generate audio file
        output_path = generate_audio(request.text)
        
        # Verify file exists before responding
        if not os.path.exists(output_path):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Audio file generation failed"
            )
        
        # Add cleanup task and return response
        background_tasks.add_task(cleanup_resources, output_path)
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="generated_audio.wav"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Cleanup immediately if error occurs
        if output_path and os.path.exists(output_path):
            os.remove(output_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS processing failed: {str(e)}"
        )

def cleanup_resources(path: str):
    """Clean up generated files after response is sent"""
    try:
        if path and os.path.exists(path):
            os.remove(path)
            print(f"Cleaned up temporary file: {path}")
    except Exception as e:
        print(f"Error cleaning up files: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
