import whisper
import torch
import numpy as np





DEVICE = "cpu"

def load_model(MODEL_TYPE: str = "base.en"):
    #torch.cuda.is_available()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(MODEL_TYPE, device=DEVICE)
    print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )
    return model



def transcribe_audio(model, audio_file: str):
    file_name = audio_file#"audio/1_voice_chunk.wav"
    result = model.transcribe(file_name)
    output_text = result["text"]
    return output_text