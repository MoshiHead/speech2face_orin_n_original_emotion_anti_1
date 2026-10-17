"""tools/offline_capture.py -- stream audio to the server and capture video frames.

    python tools/offline_capture.py <port> <audio.wav> <output_dir> [extra_silence_seconds]

Connects to the running server, streams the WAV file in real time, captures all
video JPEG frames into <output_dir>/frames/, and assembles them into an MP4.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time

import numpy as np
import torchaudio
import websockets

PORT = sys.argv[1] if len(sys.argv) > 1 else "8991"
AUDIO = sys.argv[2] if len(sys.argv) > 2 else "tools/test_question.wav"
OUTDIR = sys.argv[3] if len(sys.argv) > 3 else "offline_test"
EXTRA_SILENCE = float(sys.argv[4]) if len(sys.argv) > 4 else 15.0
SR, FRAME = 24000, 1920

wav, sr = torchaudio.load(AUDIO)
wav = wav.mean(0) if wav.shape[0] > 1 else wav[0]
if sr != SR:
    wav = torchaudio.functional.resample(wav, sr, SR)
p = wav.numpy()
audio_sec = len(p) / SR
total_sec = audio_sec + EXTRA_SILENCE
need = int(total_sec * SR)
p = np.concatenate([p, 1e-4 * np.random.randn(max(0, need - len(p))).astype(np.float32)])[:need]
i16 = (np.clip(p, -1, 1) * 32767).astype(np.int16)

frame_dir = os.path.join(OUTDIR, "frames")
os.makedirs(frame_dir, exist_ok=True)


async def main():
    cnt = {"a": 0, "v": 0}
    frames = []
    ws = await websockets.connect(
        f"ws://127.0.0.1:{PORT}/ws/conversation",
        max_size=None,
        ping_interval=None,
    )
    ready = json.loads(await ws.recv())
    print(f"Session ready: {ready}")
    vws = await websockets.connect(
        f"ws://127.0.0.1:{PORT}{ready.get('video_ws_path')}",
        max_size=None,
        ping_interval=None,
    )

    async def rx_audio(sock):
        try:
            async for _ in sock:
                cnt["a"] += 1
        except Exception:
            pass

    async def rx_video(sock):
        try:
            async for msg in sock:
                cnt["v"] += 1
                if isinstance(msg, bytes) and len(msg) > 100:
                    frames.append(msg)
                    fn = os.path.join(frame_dir, f"{len(frames):06d}.jpg")
                    with open(fn, "wb") as f:
                        f.write(msg)
        except Exception:
            pass

    tasks = [asyncio.create_task(rx_audio(ws)), asyncio.create_task(rx_video(vws))]
    await ws.send(json.dumps({"type": "start", "sample_rate": SR}))
    await asyncio.sleep(1.0)

    t0 = time.perf_counter()
    n = len(i16) // FRAME
    print(f"Streaming {audio_sec:.1f}s of audio + {EXTRA_SILENCE:.1f}s silence = {total_sec:.1f}s total ...")
    for i in range(n):
        await ws.send(b"\x03" + i16[i * FRAME : (i + 1) * FRAME].tobytes())
        dt = t0 + (i + 1) * FRAME / SR - time.perf_counter()
        if dt > 0:
            await asyncio.sleep(dt)
        if (i + 1) % 150 == 0:
            print(f"  ... {(i+1)*FRAME/SR:.0f}s sent, {cnt['v']} frames captured", flush=True)

    await asyncio.sleep(5)
    for t in tasks:
        t.cancel()
    for s in (ws, vws):
        try:
            await s.close()
        except Exception:
            pass
    return cnt, frames, time.perf_counter() - t0


def assemble_video(frames_count: int, outdir: str, audio_file: str):
    mp4 = os.path.join(outdir, "output.mp4")
    if frames_count > 10:
        fdir = os.path.join(outdir, "frames")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "25",
            "-i",
            os.path.join(fdir, "%06d.jpg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            mp4,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Video assembled: {mp4} ({os.path.getsize(mp4)/1e6:.1f} MB)")
        except Exception as e:
            print(f"ffmpeg failed: {e}")
    else:
        print("Too few frames captured to assemble a video.")


if __name__ == "__main__":
    cnt, frames, el = asyncio.run(main())
    print(f"\nDone: {cnt['a']} audio msgs, {cnt['v']} video frames ({cnt['v']/max(el,1):.1f} fps)")
    print(f"Saved {len(frames)} JPEG frames to {frame_dir}/")
    assemble_video(len(frames), OUTDIR, AUDIO)
