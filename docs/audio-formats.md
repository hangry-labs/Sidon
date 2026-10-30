# Audio upload and conversion

Use **Convert** to change formats without loading or running an inference model.
All four workflows share the same output-format selector, downloads and ZIP
manifests. Conversion jobs use the existing queue, authentication, upload/duration
limits, deletion and expiry rules.

| Download | Encoding | Sample rate |
| --- | --- | --- |
| `wav` | Float32 PCM | Workflow rate |
| `flac` | 24-bit PCM | Workflow rate |
| `opus` | Opus, 96 kbps VBR | 48 kHz |
| `mp3` | MP3, 192 kbps | 48 kHz |
| `m4a` | AAC, 192 kbps, MPEG-4 container | 48 kHz |
| `aac` | AAC, 192 kbps, ADTS container | 48 kHz |
| `ogg` | Vorbis, quality 5 | 48 kHz |

Conversion keeps channels, with MP3 downmixed to at most stereo. WAV/FLAC retain
the input sample rate. Enhancement resynthesizes mono at 48 kHz; separation
produces two mono tracks at 16 kHz before export. Compressed export resamples
each track to 48 kHz. Export does not apply loudness normalization. Float WAV
preserves peaks above full scale; FLAC clips them. Compressed encodings are lossy
and may add codec padding or slightly change decoded duration/peaks. Manifest
duration describes the source timeline before encoding.

Uploads are inspected by content, not filename. libsndfile handles its supported
recordings; the FFmpeg fallback adds AAC/M4A, WebM audio and other allowed audio
containers. Playlists and network protocols are excluded. Decoding limits remain
8–192 kHz, at most eight channels and the configured duration/upload size.
Enhancement/separation downmix input; Convert preserves its channels. Browser
playback support varies by codec; files can still be converted and downloaded.

```bash
curl -X POST http://localhost:7868/api/jobs -F file=@voice.m4a -F mode=convert -F output_format=opus
curl http://localhost:7868/api/jobs/JOB_ID
curl -OJ http://localhost:7868/api/jobs/JOB_ID/files/converted.opus
```

Docker includes FFmpeg/ffprobe, codec notices and matching Debian sources in the
source image. Native Python users must install FFmpeg and ffprobe on PATH. WAV
and FLAC do not require FFmpeg. No Python model dependencies were added.
