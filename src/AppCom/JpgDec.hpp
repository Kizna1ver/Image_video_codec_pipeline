#pragma once

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
#include <iomanip>

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
#include "../tool/config.hpp"
#include "../tool/ThreadBase.hpp"
#include "../tool/Timer.hpp"
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <sys/time.h>

#include "../tool/Files.hpp"

class JpgDec : public ThreadBase
{
public:
    typedef std::shared_ptr<JpgDec> Ptr;
    int myid = 0; // without use
    int gpu = 0;
    JpgDec(const int gpuid, const QS_queue &Pinput, const QS_queue &Poutput) : ThreadBase(Pinput, Poutput)
    {
        gpu = gpuid;
        third_cnt++;
        myid = third_cnt;
        State = ThreadBase::STATE::CREATE;
        Run();
    }
    ~JpgDec(){};

    struct gpujpeg_decoder *add_decoders()
    {
        struct gpujpeg_decoder *decoder = gpujpeg_decoder_create(stream);
        // 自动初始化参数
        gpujpeg_decoder_set_output_format(decoder, GPUJPEG_YUV, GPUJPEG_420_U8_P0P1P2); // key para
        // gpujpeg_decoder_output_set_cuda_buffer(&decoder_output);
        gpujpeg_decoder_output_set_default(&decoder_output);
        // decoder_output.type = GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER;//自定义CPU内存
        decoder_output.type = GPUJPEG_DECODER_OUTPUT_CUSTOM_CUDA_BUFFER; //自定义GPU内存
        // cout<<"pic quaty:"<<param.quality<<endl;
        return decoder;
    }
    struct gpujpeg_decoder *curr_decoder;
    cudaStream_t stream;
    struct gpujpeg_decoder_output decoder_output;

    void run()
    {
        ck(cuInit(0));
        cudaDeviceReset();

        CUcontext cuContext = NULL;
        CUdevice cuDevice = 0;
        ck(cuDeviceGet(&cuDevice, gpu));
        ck(cuCtxCreate(&cuContext, 0, cuDevice));
        cudaStreamCreate(&stream);
        curr_decoder = add_decoders();
        while (true)
        {
            int fg = 0xff;
            bool ret;
            switch (State)
            {
            case ThreadBase::WAIT:
                this_thread::sleep_for(chrono::milliseconds(10));
                continue;
            case ThreadBase::CREATE:
                // LOG(INFO) << "  图片线程就绪!";
                State = ThreadBase::WAIT;
                continue;
            case ThreadBase::FINISH:
                gpujpeg_decoder_destroy(curr_decoder);
                curr_decoder = add_decoders();
                // LOG(INFO) << "  图片转换结束!" << hd_cnt;
                State = ThreadBase::CREATE;
                break;
            case ThreadBase::EXIT:
                gpujpeg_decoder_destroy(curr_decoder);
                // cudaDeviceReset();
                // LOG(INFO) << "   图片转换线程退出!";
                return;
            case ThreadBase::INIT:
                gpujpeg_decoder_destroy(curr_decoder);
                curr_decoder = add_decoders();
                State = ThreadBase::RUNNING;
                break;
            case ThreadBase::RUNNING:
            {
                shared_ptr<ImageStr> pImageStr;
                // cout << "quesize: " << input->Size() << endl;
                ret = input->tryPop(pImageStr, fg);
                // cout << "ret:" << ret << "fg:" << fg << endl;
                if (fg == ImageSyncQueue<std::shared_ptr<ImageStr>>::CREATE)
                {
                    cout << "CREATE" << endl;
                    this_thread::sleep_for(chrono::milliseconds(1));
                    continue;
                }
                if (fg == ImageSyncQueue<std::shared_ptr<ImageStr>>::FINISH)
                {
                    // cout << "FINISH" << endl;
                    State = ThreadBase::CREATE;
                    continue;
                }
                if (!ret)
                {
                    break;
                }
                Timer temp_timer;
                temp_timer.Start();
                int all_cnt = pImageStr->message_len; // all_cnt <= 10
                hd_cnt += all_cnt;
                pImageStr->message_len = 0;
                for (int i = 0; i < all_cnt; i++)
                {
                    if (!pImageStr->b[i].isvalue)
                    {
                        continue;
                    }
                    decoder_output.data = pImageStr->b[i].d_pImg;
                    ck(gpujpeg_decoder_decode(curr_decoder, pImageStr->b[i].pImg, pImageStr->b[i].size, &decoder_output));
                    // if (hd_cnt % 1000 == 0)
                    delete pImageStr->b[i].pImg;
                    pImageStr->b[i].size = decoder_output.data_size;
                    pImageStr->b[i].d_pImg = decoder_output.data;
                    pImageStr->b[i].height = curr_decoder->reader->param_image.height;
                    pImageStr->b[i].width = curr_decoder->reader->param_image.width;
                    pImageStr->message_len++;
                }
                if (pImageStr->message_len != 0)
                {
                    // if (hd_cnt % 1000 == 0)
                    //     printf("JpgDec:cnt=%ld,JPG Queue Size:%ld,YUV Queue Size=%ld\n", hd_cnt, input->Size(), output->Size());
                    output->Push(pImageStr);
                }
                Handle_time += temp_timer.ElapsedSeconds();
                // double temp_cost = end - start;
                // printf("Frame encode costs: %lf\n", Handle_time);
                break;
            }
            default:
                break;
            }

            // if (hd_cnt % 100 == 0)
            // printf("%d Frame:cnt=%ld,in=%ld,out=%ld\n", myid, hd_cnt, input->Size(), output->Size());
        }
    }
};