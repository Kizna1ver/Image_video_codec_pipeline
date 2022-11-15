
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
#include "../tool/config.hpp"
#include "../tool/ThreadBase.hpp"
#include "../tool/Timer.hpp"
#include <ctime>
#include <iostream>
#include <thread>
#include <sys/time.h>

#include "../tool/Files.hpp"

class VideoEnc : public ThreadBase
{
    class Encoders
    {
    public:
        media_info minfo; // dat文件的描述性信息

        NvEncoderInitParam encodeCLIOptions;
        NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_IYUV;
        NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
        NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};

        std::unique_ptr<NvEncoderCuda> pEnc;
        cudaStream_t stream;
        std::vector<std::vector<uint8_t>> vPacket;
        std::ofstream *fpOutfile;
        string key;
        string path;

        int fram_cnt_ = 0;
        uint64_t file_size = 0;
        int cntr = 0;
        Encoders(Config config, string topic, CUcontext &cuContext)
        {
            pEnc = std::make_unique<NvEncoderCuda>(cuContext, config.image.width, config.image.height, eFormat);

            initializeParams.encodeConfig = &encodeConfig;
            initializeParams.encodeGUID == NV_ENC_CODEC_HEVC_GUID;
            pEnc->CreateDefaultEncoderParams(&initializeParams, encodeCLIOptions.GetEncodeGUID(), encodeCLIOptions.GetPresetGUID(), encodeCLIOptions.GetTuningInfo());
            initializeParams.frameRateNum = config.video.fps;
            encodeCLIOptions.SetInitParams(&initializeParams, eFormat);
            pEnc->CreateEncoder(&initializeParams);
            // config.print();
            path = config.image.output_file_name;
            // key = std::to_string(config.image.width) + "x" + std::to_string(config.image.height) + "-" + Timer::GetUTCString();
            key = topic.substr(1) + "_" + std::to_string(config.image.height) + "p";
            // cout << "key" << key << endl;
            string output_file_name = key + ".hevc";
            fpOutfile = new std::ofstream((path + output_file_name).c_str(), ios::out | std::ios::app | std::ios::binary);
            if (!fpOutfile)
            {
                std::ostringstream err;
                err << "Unable to open output file: " << output_file_name << std::endl;
                throw std::invalid_argument(err.str());
            }
            minfo.data.height = config.image.height;
            minfo.data.width = config.image.width;
        }
        int addframe(void *pSrcFrame, CUcontext &cuContext)
        {
            const NvEncInputFrame *encoderInputFrame = pEnc->GetNextInputFrame();
            NvEncoderCuda::CopyToDeviceFrame(cuContext, pSrcFrame, 0, (CUdeviceptr)encoderInputFrame->inputPtr,
                                             (int)encoderInputFrame->pitch,
                                             pEnc->GetEncodeWidth(),
                                             pEnc->GetEncodeHeight(),
                                             CU_MEMORYTYPE_DEVICE, //CU_MEMORYTYPE_HOST CU_MEMORYTYPE_DEVICE
                                             encoderInputFrame->bufferFormat,
                                             encoderInputFrame->chromaOffsets,
                                             encoderInputFrame->numChromaPlanes);
            pEnc->EncodeFrame(vPacket);
            fram_cnt_ = 0;
            for (std::vector<uint8_t> &packet : vPacket)
            {
                fpOutfile->write(reinterpret_cast<char *>(packet.data()), packet.size());
                file_size += packet.size();
                fram_cnt_++;
                minfo.frame_base.push_back(file_size); //
            }

            return fram_cnt_;
        }

        int end()
        {
            // 处理残余帧？
            fram_cnt_ = 0;
            pEnc->EndEncode(vPacket);
            for (std::vector<uint8_t> &packet : vPacket)
            {
                // printf("asd\n");
                fpOutfile->write(reinterpret_cast<char *>(packet.data()), packet.size());
                file_size += packet.size();
                minfo.frame_base.push_back(file_size); //

                fram_cnt_++;
            }
            fpOutfile->close();
            pEnc->DestroyEncoder();
            minfo.save(path, key);
            printf("Video code finished,file size:%ld MiB Frame:%d,%s\n", file_size >> 20, fram_cnt_, key.c_str());
            pEnc.release();
            return fram_cnt_;
        }
    };

public:
    CUcontext cuContext;
    Config configparam;
    static atomic<int> frame_cnt;
    static atomic<uint64_t> file_size;

    VideoEnc(Config &para, const QS_queue &Pinput,
             const QS_queue &Poutput) : ThreadBase(Pinput, Poutput)
    {
        configparam = para;
        Run();
        // configparam->print();
    }
    ~VideoEnc(){};

private:
    std::vector<thread *> thds;
    // std::map<string, Encoders *> decoders;
    void run()
    {
        ck(cuInit(0));

        cudaDeviceReset();

        CUdevice cuDevice = 0;
        ck(cuDeviceGet(&cuDevice, configparam.gpuid));
        ck(cuCtxCreate(&cuContext, 0, cuDevice));

        shared_ptr<Encoders> curr_decoder; //decoder？ encoder！！
        while (true)
        {
            switch (State)
            {
            case ThreadBase::WAIT:
                this_thread::sleep_for(chrono::milliseconds(10));
                continue;
            case ThreadBase::CREATE:
                // LOG(INFO) << "  视频线程就绪!";
                State = ThreadBase::WAIT;
                continue;
            case ThreadBase::FINISH:
                LOG(INFO) << " Video code thread FINISH!";
                curr_decoder->end();
                file_size += curr_decoder->file_size;
                curr_decoder.reset();
                State = ThreadBase::CREATE;
                continue;
            case ThreadBase::EXIT:
                // LOG(INFO) << "    视频转换线程退出!";
                return;
            case ThreadBase::INIT:
                break;
            case ThreadBase::RUNNING:
            {
                shared_ptr<ImageStr> pImageStr;

                int fg = 0xff;
                bool ret = input->tryPop(pImageStr, fg);
                if (fg == ImageSyncQueue<std::shared_ptr<ImageStr>>::CREATE)
                {
                    this_thread::sleep_for(chrono::milliseconds(1));
                    continue;
                }
                if (fg == ImageSyncQueue<std::shared_ptr<ImageStr>>::FINISH)
                {
                    State = ThreadBase::FINISH;
                    // output->State = ImageSyncQueue<std::shared_ptr<ImageStr>>::FINISH;
                    // this_thread::sleep_for(chrono::milliseconds(10));
                    continue;
                }
                if (!ret)
                {
                    break;
                }
                Timer temp_timer;
                temp_timer.Start();
                if (curr_decoder.use_count() == 0)
                {
                    configparam.image.height = pImageStr->b[0].height;
                    configparam.image.width = pImageStr->b[0].width;
                    curr_decoder = make_shared<Encoders>(configparam, topic, cuContext); //= new Encoders(width, height, configparam->image.output_file_name, cuContext);
                }
                for (int i = 0; i < pImageStr->message_len; i++, hd_cnt++)
                {
                    curr_decoder->addframe(pImageStr->b[i].d_pImg, cuContext);
                    if (hd_cnt % 1000 == 0)
                        printf("VideoEnc:Frame count has been compressed:%ld\n", hd_cnt);
                    curr_decoder->minfo.timestamps.push_back(pImageStr->b[i].timestamp);
                }
                Handle_time += temp_timer.ElapsedSeconds();
                output->Push(pImageStr);
                break;
            }
            default:
                break;
            }
        }

        return;
    }
};
atomic<int> VideoEnc::frame_cnt{0};
atomic<uint64_t> VideoEnc::file_size{0};
