# image-video-codec-pipeline

## Introduction

A project to compress continuous JPG images into HEVC video stream files

## Requirements

This pipeline is based on the following two projects :

* NVIDIA VIDEO CODEC SDK(SDK:https://developer.nvidia.com/nvidia-video-codec-sdk), example code(https://github.com/NVIDIA/video-sdk-samples)
* GPUJPEG(https://github.com/CESNET/GPUJPEG)

For requirements of each project, please browse the links above.

### Linux

* Refer to the NVIDIA Video SDK developer zone web page (https://developer.nvidia.com/nvidia-video-codec-sdk) for GPUs which support video encoding and decoding acceleration.
* Driver version 455.27 or higher 
* CUDA 11.0 or higher Toolkit (http://developer.nvidia.com/cuda/cuda-toolkit)
* CMake version 3.9 or higher (https://cmake.org/download/)
* FFmpeg version 4.2.4 or higher (https://www.ffmpeg.org/download.html)

## How to build
* Install all dependencies
* Create a subfolder named "build" in the project root folder
* Use the following command to build samples in release mode.

```shell
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j14
```
