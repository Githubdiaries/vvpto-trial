# VeinVision — Live Vein Processing

Real-time camera processing prototype using Raspberry Pi Camera Module 3 NoIR,
OpenCV, FFmpeg, and MediaMTX WebRTC.

## Pipeline

Camera -> Picamera2 -> OpenCV VeinPipeline -> FFmpeg -> MediaMTX -> WebRTC

## Requirements

- Raspberry Pi 5
- Raspberry Pi Camera Module 3 NoIR
- Picamera2 and libcamera
- OpenCV and NumPy
- FFmpeg with libx264 support
- MediaMTX for the device architecture

## Run

Terminal 1:

    cd ~/vvpto-trial/mediamtx-webrtc
    ./mediamtx

Terminal 2:

    cd ~/vvpto-trial/mediamtx-webrtc
    python3 process_stream.py

Open this URL in a browser, replacing the IP with your Pi's current
Tailscale IP:

    http://<PI-TAILSCALE-IP>:8889/cam

## Processing output

The diagnostic video shows four panels:
- RAW
- ENHANCED
- VEIN MASK
- VEIN OVERLAY

## Illumination mode

The current processing script uses continuous IR illumination and
does not control GPIO 18. It does not use alternating LED-on/off
frame subtraction.

## Limitations

This is an experimental computer-vision prototype. Detected lines
have not been clinically validated as veins.
