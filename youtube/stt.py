import os
import glob
import json
import torch
import yt_dlp
import numpy as np
import noisereduce as nr
from pydub import AudioSegment, effects
from faster_whisper import WhisperModel
from config import YOUTUBE_PATH 


YOUTUBE_URL = YOUTUBE_PATH
OUTPUT_FILENAME = "whisper_transcript.json"


def download_audio_from_youtube(url: str, out_template='downloaded_audio.%(ext)s'):
    """YouTube에서 오디오를 다운로드합니다."""
    for f in glob.glob('downloaded_audio.*'):
        os.remove(f)
    ydl_opts = {
        'format': 'bestaudio/best', 'outtmpl': out_template, 'quiet': False, 'no_warnings': True,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}]
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return glob.glob('downloaded_audio.*')[0]
    except Exception as e:
        print(f"다운로드 에러: {e}")
        return None


def convert_to_wav(input_filepath: str, wav_path='converted_audio.wav'):
    """오디오 파일을 16kHz 모노 WAV로 변환하고 노이즈를 제거합니다."""
    try:
        audio = AudioSegment.from_file(input_filepath).set_channels(1).set_frame_rate(16000)
        audio = effects.normalize(audio)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        reduced_noise_samples = nr.reduce_noise(y=samples, sr=16000)
        processed_samples = (reduced_noise_samples * 32767).astype(np.int16)
        processed_audio = AudioSegment(processed_samples.tobytes(), frame_rate=16000, sample_width=2, channels=1)
        processed_audio.export(wav_path, format='wav', codec='pcm_s16le')
        return wav_path
    except Exception as e:
        print(f"WAV 변환 에러: {e}")
        return None


# 짧은 문장 자동 병합 함수
def merge_short_segments(transcripts, min_length=15):
    """길이가 짧은 문장들을 인접 문장과 병합."""
    merged = []
    buffer = ""
    start_time = None
    for seg in transcripts:
        text = seg["text"].strip()
        if not text:
            continue
        if start_time is None:
            start_time = seg["start"]
        buffer += " " + text
        #  문장 길이가 충분하거나 문장부호로 끝나면 문장 확정
        if len(buffer) > min_length or text.endswith(("다.", "요.", "죠.", "까?", "죠?")):
            merged.append({
                "text": buffer.strip(),
                "start": start_time,
                "end": seg["end"]
            })
            buffer = ""
            start_time = None
    # 마지막 문장 처리
    if buffer:
        merged.append({
            "text": buffer.strip(),
            "start": start_time if start_time else 0,
            "end": transcripts[-1]["end"]
        })
    return merged


def transcribe_with_whisper(model: WhisperModel, wav_path: str, language="ko"):
    """사전 로드된 Whisper 모델을 사용하여 오디오를 텍스트로 변환합니다."""
    try:
        # 긴 문장 단위로 인식 (chunk_length 확장)
        segments, _ = model.transcribe(
            wav_path,
            beam_size=5,
            vad_filter=False,       # 짧은 구간 자동 분할 비활성화
            chunk_length=90,        # ✅ 최대 90초 단위로 인식
            language=language
        )

        transcripts = [
            {"text": s.text.strip(), "start": round(s.start, 2), "end": round(s.end, 2)}
            for s in segments
        ]

        # 짧은 문장은 자동 병합
        transcripts = merge_short_segments(transcripts, min_length=20)

        return transcripts
    except Exception as e:
        print(f"Whisper 인식 에러: {e}")
        return None


