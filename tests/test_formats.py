# SPDX-License-Identifier: Apache-2.0
import io
import json
from pathlib import Path
import shutil
import time
import zipfile

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from sidon.web.app import create_app
from sidon.web.audio import decode_input, encode_audio


def completed(client, job_id):
    for _ in range(500):
        job = client.get(f'/api/jobs/{job_id}').json()
        if job['status'] in {'done', 'failed'}:
            assert job['status'] == 'done', job
            return job
        time.sleep(.01)
    pytest.fail('Conversion job did not finish')


@pytest.mark.parametrize('format', ['wav', 'flac', 'opus', 'mp3', 'm4a', 'aac', 'ogg'])
def test_conversion_download_and_reupload_without_loading_models(tmp_path, monkeypatch, format):
    if format not in {'wav', 'flac'} and not shutil.which('ffmpeg'):
        pytest.skip('FFmpeg is required for compressed codecs')
    monkeypatch.setattr('sidon.web.app.Runtime.load', lambda *a: pytest.fail('Conversion must not load a model'))
    timebase = np.arange(22050) / 44100
    source = np.stack([.2*np.sin(timebase*1000), .1*np.cos(timebase*2000)], axis=1).astype('float32')
    stream = io.BytesIO()
    sf.write(stream, source, 44100, format='WAV', subtype='FLOAT')
    app = create_app(tmp_path)
    with TestClient(app) as client:
        response = client.post('/api/jobs', files={'file': ('input.bin', stream.getvalue())}, data={'mode':'convert', 'output_format':format})
        assert response.status_code == 202, response.text
        job = completed(client, response.json()['id'])
        assert job['backends'] == {} and app.state.runtime.models == {}
        track = job['outputs'][0]
        assert track['channels'] == 2
        assert track['sample_rate'] == (44100 if format in {'wav','flac'} else 48000)
        assert track['lossy'] == (format not in {'wav','flac'})
        encoded = client.get(f"/api/jobs/{job['id']}/files/{track['file']}").content
        archive = zipfile.ZipFile(io.BytesIO(client.get(f"/api/jobs/{job['id']}/files/results.zip").content))
        assert json.loads(archive.read('run.json'))['outputs'] == job['outputs']
        assert archive.read(track['file']) == encoded
        reupload = client.post('/api/jobs', files={'file': ('misleading.data', encoded)}, data={'mode':'convert', 'output_format':'wav'})
        assert reupload.status_code == 202, reupload.text
        again = completed(client, reupload.json()['id'])
        wave = client.get(f"/api/jobs/{again['id']}/files/converted.wav").content
        actual, rate = sf.read(io.BytesIO(wave), dtype='float32')
        assert actual.shape[1] == 2 and np.isfinite(actual).all()
        assert abs(len(actual)/rate - .5) < .08
        assert np.sqrt(np.mean(actual**2)) > .05
        if format == 'wav':
            np.testing.assert_array_equal(source, actual)


def test_lossless_peak_policy(tmp_path):
    x = np.array([0, 1.2, -1.2, .5], dtype='float32')
    track = encode_audio(x, 16000, tmp_path/'audio.wav', 'wav')
    assert track['samples_over_full_scale'] == 2
    np.testing.assert_array_equal(sf.read(tmp_path/'audio.wav', dtype='float32')[0], x)
    encode_audio(x, 16000, tmp_path/'audio.flac', 'flac')
    assert np.max(np.abs(sf.read(tmp_path/'audio.flac')[0])) <= 1


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg required')
def test_multichannel_mp3_downmix_is_reported(tmp_path):
    x = np.tile((.1*np.sin(np.arange(16000)*.1))[:,None], (1,6)).astype('float32')
    track = encode_audio(x, 16000, tmp_path/'audio.mp3', 'mp3')
    assert track['channels'] == 2 and track['sample_rate'] == 48000
    decode_input(tmp_path/'audio.mp3', tmp_path/'input.wav', 2, preserve_channels=True)
    assert sf.info(tmp_path/'input.wav').channels == 2


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg required')
def test_playlist_input_is_rejected(tmp_path):
    secret = tmp_path/'secret.wav'
    sf.write(secret, np.ones(1600, dtype='float32'), 16000)
    playlist = tmp_path/'upload'
    playlist.write_text("ffconcat version 1.0\nfile 'secret.wav'\n")
    with pytest.raises(ValueError):
        decode_input(playlist, tmp_path/'input.wav', 2)
    assert not (tmp_path/'input.wav').exists()
