from __future__ import annotations

import os
from loguru import logger
import torch
import whisperx


def transcribe_and_translate(
    input_mp4: str,
    output_srt: str,
    language: str = "auto",
    model_size: str = "medium",
) -> None:
    os.makedirs(os.path.dirname(output_srt), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    logger.info("Loading WhisperX model: {} on {}", model_size, device)
    model = whisperx.load_model(model_size, device, compute_type=compute_type)

    logger.info("Transcribing {}", input_mp4)
    audio = whisperx.load_audio(input_mp4)
    result = model.transcribe(audio, language=None if language == "auto" else language)

    # Align for better word timings
    try:
        model_a, metadata = whisperx.load_align_model(
            language=result.get("language", "en"), device=device
        )
        result = whisperx.align(result["segments"], model_a, metadata, audio, device)
    except Exception:
        pass

    # If not English, translate to English
    segments = result.get("segments") if isinstance(result, dict) else result
    if result.get("language") != "en":
        logger.info("Detected language {}, translating to English", result.get("language"))
        # Use translate=True path via base Whisper model for translation text
        trans_model = whisperx.load_model(model_size, device, compute_type=compute_type)
        tr = trans_model.transcribe(audio, language="en", task="translate")
        segments = tr.get("segments")

    # Write SRT
    with open(output_srt, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            start = seg["start"]
            end = seg["end"]
            text = seg["text"].strip()
            f.write(f"{idx}\n")
            f.write(f"{_srt_ts(start)} --> {_srt_ts(end)}\n")
            f.write(f"{text}\n\n")


def _srt_ts(t: float) -> str:
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    milliseconds = int((t - int(t)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


