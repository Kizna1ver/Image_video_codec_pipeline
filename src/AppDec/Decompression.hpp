#pragma once
#ifndef __VIDEODEC__H
#define __VIDEODEC__H
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "gpujpeg_common.h"
#include "gpujpeg_encoder.h"
#include "gpujpeg_type.h"
#include "gpujpeg_decoder.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thread>
#include "./nvencoder/NvEncoderCuda.h"
#include "./nvdecoder/NvDecoder.h"
#include "nvutils/Logger.h"
#include "nvutils/NvCodecUtils.h"
#include "nvutils/FFmpegDemuxer.h"
#include "nvutils/NvEncoderCLIOptions.h"
#include "./gpujpeg_decoder_internal.h"
#include <atomic>
#include "../tool/ImageSyncQueue.hpp"
#include <queue>
#include "stdio.h"
#include <unistd.h>

#include <ctime>
#include <iostream>
#include <thread>
#include <sys/time.h>

#include "../tool/Files.hpp"
#include "../tool/config.hpp"
#include "../tool/ThreadBase.hpp"
#include "../tool/Timer.hpp"

class Decompression : public ThreadBase
{
public:
    static atomic<int> frame_cnt;
    static atomic<int> file_size;
    const char *szOutFilePath;
    media_info *m2;
    Decompression(const Config &param, media_info &m2_) : ThreadBase(nullptr, nullptr)
    {
        file_name = param.image.input_file_name; // "data4/nasa_1280/";
        szOutFilePath = param.image.output_file_name.c_str();
        gpu = param.gpuid;
        this->m2 = &m2_;
    }
    ~Decompression(){
        // printf("～Prod/uce_images,%d\n", __LINE__);
    };

    template <class EncoderClass>
    void InitializeEncoder(EncoderClass &pEnc, NvEncoderInitParam encodeCLIOptions, NV_ENC_BUFFER_FORMAT eFormat)
    {
        NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
        NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};

        initializeParams.encodeConfig = &encodeConfig;
        pEnc->CreateDefaultEncoderParams(&initializeParams, encodeCLIOptions.GetEncodeGUID(), encodeCLIOptions.GetPresetGUID(), encodeCLIOptions.GetTuningInfo());
        encodeCLIOptions.SetInitParams(&initializeParams, eFormat);

        pEnc->CreateEncoder(&initializeParams);
    }

    void ConvertSemiplanarToPlanar(uint8_t *pHostFrame, int nWidth, int nHeight, int nBitDepth)
    {
        if (nBitDepth == 8)
        {
            // nv12->iyuv
            YuvConverter<uint8_t> converter8(nWidth, nHeight);
            converter8.UVInterleavedToPlanar(pHostFrame);
        }
        else
        {
            // p016->yuv420p16
            YuvConverter<uint16_t> converter16(nWidth, nHeight);
            converter16.UVInterleavedToPlanar((uint16_t *)pHostFrame);
        }
    }

    string file_name;
    int gpu;
    int image_num;
    string folder_name;
    void run()
    {
        ck(cuInit(0));
        CUcontext cuContext = NULL;
        CUdevice cuDevice = 0;
        ck(cuDeviceGet(&cuDevice, gpu));
        ck(cuCtxCreate(&cuContext, 0, cuDevice));

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // hevc2yuv
        const unsigned int opPoint = 0;
        const bool bDispAllLayers = true;

        int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
        uint8_t *p_video = NULL, *p_frame;

        NvDecoder::Rect cropRect = {};
        Dim resizeDim = {};

        FFmpegDemuxer demuxer(file_name.c_str()); //"/home/tusimple/Videos/output/xxxx.hevc"
        // std::cout << "RUN" << file_name << endl;
        // 第二个参数表示是否使用显存，可优化。
        NvDecoder dec(cuContext, false, FFmpeg2NvCodecId(demuxer.GetVideoCodec()), false, false, &cropRect, &resizeDim);
        dec.SetOperatingPoint(opPoint, bDispAllLayers);
        LOG(INFO) << dec.GetVideoInfo();
        int frame_cnt_ = 0;

        int image_compressed_size = 0;
        uint8_t *image_compressed = NULL;
        char output_file_name[256] = "";

        // yuv2jpg
        struct gpujpeg_parameters param;
        struct gpujpeg_image_parameters param_image;
        struct gpujpeg_encoder_input encoder_input;

        gpujpeg_set_default_parameters(&param);

        param_image.comp_count = 3;
        param_image.color_space = GPUJPEG_YUV;
        param_image.pixel_format = GPUJPEG_420_U8_P0P1P2;

        struct gpujpeg_encoder *encoder = gpujpeg_encoder_create(stream);
        int cnt = 0;

        int h_cnt = 0;

        do
        {
            // 解码多帧
            // cuvidReconfigureDecoder();
            // Demux 解析,获得每一帧码流的数据存在pVideo中,nVideoBytes为数据的字节数
            // Demux将pVideo存储的地址值改变为pkt.data，即改变了pVideo指向的地址！！！
            demuxer.Demux(&p_video, &nVideoBytes);
            nFrameReturned = dec.Decode(p_video, nVideoBytes); //实际解码进入函数
            for (int i = 0; i < nFrameReturned; i++)
            {
                p_frame = dec.GetFrame();

                // 图片格式转换？
                ConvertSemiplanarToPlanar(p_frame, dec.GetWidth(), dec.GetHeight(), dec.GetBitDepth());

                param_image.height = dec.GetHeight();
                param_image.width = dec.GetWidth();

                gpujpeg_encoder_input_set_image(&encoder_input, p_frame); // pFrame is yuv image ptr
                if (gpujpeg_encoder_encode(encoder, &param, &param_image, &encoder_input,
                                           &image_compressed, &image_compressed_size) != 0)
                {
                    printf("gpujpeg_encoder_encode error %d\n", __LINE__);
                    continue;
                }

                sprintf(output_file_name, "%s/%ld.jpg", szOutFilePath, m2->timestamps[frame_cnt_]);
                cnt++;
                std::ofstream fpOut(output_file_name, std::ios::out | std::ios::binary);
                h_cnt++;
                // // write jpg file
                // printf("%s,%d\n", output_file_name, image_compressed_size);
                fpOut.write(reinterpret_cast<char *>(image_compressed), image_compressed_size);
                fpOut.close();
                frame_cnt_++;

                // delete image_compressed;
            }
            // printf("in:%d\n", nFrame);
            nFrame += nFrameReturned;
        } while (nVideoBytes);
        // delete p_video;  segment_fault
        // dec.setReconfigParams(&cropRect, &resizeDim);
        frame_cnt += frame_cnt_;
        cudaDeviceReset();                // 2M显存
        ck(cudaStreamDestroy(stream));    // 几乎不占显存
        gpujpeg_encoder_destroy(encoder); // 80M显存
        // cuCtxDestroy(cuContext);          // 完全销毁上下文,似乎和NvDoder的析构有冲突，corrupted size vs. prev_size，但是不销毁又会泄漏显存。
        cout << szOutFilePath << " finished,total count:" << h_cnt << endl;
        // std::cout << "get frame end" << std::endl;
        return;
    }
};
atomic<int> Decompression::frame_cnt{0};

#endif