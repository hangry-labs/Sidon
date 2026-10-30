# SPDX-License-Identifier: Apache-2.0
"""Bounded upload decoding and shared export encoding, independent of models."""
from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess
from typing import Literal

from fastapi import HTTPException
import numpy as np
import soundfile as sf

OutputFormat = Literal['wav', 'flac', 'opus', 'mp3', 'm4a', 'aac', 'ogg']
CODECS = {
    'opus': ['-c:a', 'libopus', '-b:a', '96k', '-vbr', 'on', '-application', 'audio'],
    'mp3': ['-c:a', 'libmp3lame', '-b:a', '192k'],
    'm4a': ['-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart'],
    'aac': ['-c:a', 'aac', '-b:a', '192k'],
    'ogg': ['-c:a', 'libvorbis', '-q:a', '5'],
}
# Reject playlists and external-reference demuxers rather than relying on names.
INPUT_FORMATS = 'wav,flac,ogg,mp3,aac,mov,aiff,matroska,webm,asf,au,caf'


def executable(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError('Install FFmpeg and ffprobe for compressed audio conversion')
    return path


def run(command: list[str], timeout=120):
    try:
        return subprocess.run(command, capture_output=True, check=True, timeout=timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError('Audio conversion failed or exceeded its time limit') from exc


def decode_input(source: Path, destination: Path, max_seconds: int, *, preserve_channels=False) -> float:
    """Decode in blocks, limiting duration/rate/channels even with false metadata."""
    temporary = destination.with_name('decoded.wav')
    try:
        try:
            stream = sf.SoundFile(source)
        except (RuntimeError, sf.LibsndfileError):
            # Uploaded names never become command arguments or select a decoder.
            probe = run([executable('ffprobe'), '-v', 'error', '-protocol_whitelist', 'file,pipe',
                         '-format_whitelist', INPUT_FORMATS, '-show_entries',
                         'stream=codec_type,sample_rate,channels', '-of', 'json', str(source)], timeout=15)
            tracks = [x for x in json.loads(probe.stdout)['streams'] if x.get('codec_type') == 'audio']
            if not tracks:
                raise ValueError('No audio stream')
            rate, channels = int(tracks[0]['sample_rate']), int(tracks[0]['channels'])
            if not 8000 <= rate <= 192000 or not 1 <= channels <= 8:
                raise ValueError('Unsupported audio rate or channel count')
            run([executable('ffmpeg'), '-nostdin', '-v', 'error', '-y', '-threads', '2',
                 '-protocol_whitelist', 'file,pipe', '-format_whitelist', INPUT_FORMATS,
                 '-i', str(source), '-map', '0:a:0', '-vn', '-t', str(max_seconds + 1),
                 '-c:a', 'pcm_f32le', str(temporary)])
            stream = sf.SoundFile(temporary)
        with stream:
            if stream.frames <= 0 or not 8000 <= stream.samplerate <= 192000 or not 1 <= stream.channels <= 8 or stream.frames / stream.samplerate > max_seconds:
                raise HTTPException(422, 'Use a nonempty clip within the duration limit, 8–192 kHz, and at most 8 channels')
            total = 0
            with sf.SoundFile(destination, 'w', samplerate=stream.samplerate,
                              channels=stream.channels if preserve_channels else 1, subtype='FLOAT', format='WAV') as output:
                for block in stream.blocks(blocksize=65536, dtype='float32', always_2d=True):
                    total += len(block)
                    if total > max_seconds * stream.samplerate or not np.isfinite(block).all():
                        raise HTTPException(422, 'Audio exceeds the duration limit or contains nonfinite samples')
                    output.write(block if preserve_channels else block.mean(axis=1))
            if not total:
                raise HTTPException(422, 'The audio file is empty')
            return total / stream.samplerate
    finally:
        temporary.unlink(missing_ok=True)


def encode_file(source: Path, destination: Path, output_format: OutputFormat) -> dict:
    """Stream lossless conversion and use bounded external codec processes."""
    if output_format not in {'wav', 'flac', *CODECS}:
        raise ValueError('Unsupported output format')
    clipped = 0
    with sf.SoundFile(source) as stream:
        rate, channels, frames = stream.samplerate, stream.channels, stream.frames
        output = sf.SoundFile(destination, 'w', samplerate=rate, channels=channels,
                              format=output_format.upper(), subtype='FLOAT' if output_format == 'wav' else 'PCM_24') if output_format in {'wav', 'flac'} else None
        try:
            for block in stream.blocks(blocksize=65536, dtype='float32', always_2d=True):
                if not np.isfinite(block).all():
                    raise ValueError('Audio contains nonfinite samples')
                clipped += int(np.count_nonzero(np.abs(block) > 1))
                if output:
                    output.write(block)
        finally:
            if output:
                output.close()
    if output_format in CODECS:
        if output_format == 'mp3':
            channels = min(channels, 2)
        run([executable('ffmpeg'), '-nostdin', '-v', 'error', '-y', '-threads', '2',
             '-i', str(source), '-map_metadata', '-1', '-map', '0:a:0', '-vn', '-ar', '48000', '-ac', str(channels),
             *CODECS[output_format], str(destination)])
        rate = 48000
    return dict(file=destination.name, sample_rate=rate, channels=channels,
                duration=frames / sf.info(source).samplerate, samples_over_full_scale=clipped,
                format=output_format, lossy=output_format in CODECS)


def encode_audio(data: np.ndarray, rate: int, destination: Path, output_format: OutputFormat) -> dict:
    # Keep the default inference exports direct, without an extra disk roundtrip.
    if output_format in {'wav', 'flac'}:
        if not np.isfinite(data).all():
            raise ValueError('Audio contains nonfinite samples')
        sf.write(destination, data, rate, subtype='FLOAT' if output_format == 'wav' else 'PCM_24')
        return dict(file=destination.name, sample_rate=rate, channels=1 if data.ndim == 1 else data.shape[1],
                    duration=len(data) / rate, samples_over_full_scale=int(np.count_nonzero(np.abs(data) > 1)),
                    format=output_format, lossy=False)
    temporary = destination.with_suffix('.source.wav')
    try:
        sf.write(temporary, data, rate, subtype='FLOAT')
        return encode_file(temporary, destination, output_format)
    finally:
        temporary.unlink(missing_ok=True)
