#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "gpujpeg_common.h"

#include "gpujpeg_encoder.h"
#include "gpujpeg_type.h"
#include "gpujpeg_common.h"
#include "gpujpeg_decoder.h"

#include "nvencoder/NvEncoderCuda.h"
#include "nvdecoder/NvDecoder.h"

#include "nvutils/Logger.h"
#include "nvutils/NvCodecUtils.h"
#include "nvutils/FFmpegDemuxer.h"
#include "nvutils/NvEncoderCLIOptions.h"
#include "gpujpeg_decoder_internal.h"

#include <atomic>
#include "../tool/ImageSyncQueue.hpp"
#include <queue>
#include "stdio.h"
#include <unistd.h>
#include "../tool/config.hpp"
#include "../tool/ThreadBase.hpp"
#include "../tool/Timer.hpp"
#include "../tool/ThreadPool.hpp"
#include "../tool/Files.hpp"

#include <iostream>
#include <sys/time.h>
#include "Decompression.hpp"

void Decompression_task(media_info &m2, string infilename, string outdfile)
{
    Config configparam;
    configparam.gpuid = 0;
    configparam.image.width = m2.data.width;
    configparam.image.height = m2.data.height;
    configparam.image.input_file_name = infilename; // "/home/tusimple/Videos/output/xxxx.hevc"
    configparam.image.output_file_name = outdfile;

    // cout << "---------------" << outdfile << endl;
    unique_ptr<Decompression> produce_thds;
    produce_thds = std::make_unique<Decompression>(configparam, m2); // 不需要内部工作队列，整个是串行的pipeline
    produce_thds->run();
}

void toDoTask(string input_path, int pool_size)
{
    Timer devinit;
    clock_t start = 0;
    clock_t end = 0;

    start = clock();
    devinit.Start();

    std::vector<std::string> mFileVec;
    scan_files(mFileVec, input_path); // 获取.dat 文件路径
    assert(mFileVec.size() != 0);
    ThreadPool tp(pool_size, 25);
    for (auto filename : mFileVec)
    {
        media_info *m2 = new media_info();
        m2->read(input_path + filename); // 从xxxx.dat读取信息到m2里
        mkdir((input_path + m2->data.filename).c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
        tp.commit(Decompression_task, *m2, input_path + m2->data.filename + ".hevc", input_path + m2->data.filename);
    }
    this_thread::sleep_for(chrono::milliseconds(3000));
    while (true)
    {
        if (tp.idlCount() == 20)
            break;
        std::this_thread::yield();
    }
    end = clock();
    double real_time = devinit.ElapsedSeconds();
    double use_cpu_time = (double)(end - start) / CLOCKS_PER_SEC; /*#define CLOCKS_PER_SEC ((clock_t)1000)*/

    cout.setf(ios::fixed);
    LOG(INFO) << "\tExecute Time:" << real_time << setprecision(3) << " 秒,"
              << "\tCPU Avarage Use:(" << (int)(use_cpu_time / real_time * 100) << " %/ 1200 %)";
}

int dec_pool(int argc, char **argv)
{
    vector<Config> configs;
    read_config(configs);
    assert(configs.size() != 0);
    configs[0].print();
    toDoTask(configs[0].image.output_file_name, configs[0].dec_pool_size); // "/home/tusimple/Videos/output/"
    return 0;
}
