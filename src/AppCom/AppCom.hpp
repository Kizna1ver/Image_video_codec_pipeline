
#include <string.h>
#include <iomanip>
#include <future> // std::promise, std::future
#include <time.h>
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

#include "../tool/config.hpp"
#include "../tool/Timer.hpp"
#include "../tool/Files.hpp"
#include "../tool/ImageSyncQueue.hpp"

#include "../AppCom/JpgDec.hpp"
#include "../AppCom/VideoEnc.hpp"
#include "../AppCom/ReadPictureFromFile.hpp"
#include "../AppCom/SavePictureFromDataSet.hpp"

class StateMachine
{
public:
    QS_queue mq_jpg = CreatQS_queue(2048);       //存放jpg的队列，由Read线程Push JpgDec Pop
    QS_queue mq_yuv = CreatQS_queue(2048);       //存放YUV的队列，由JpgDec线程Push VideoEnc Pop
    QS_queue mq_empty_box = CreatQS_queue(2048); //存放空消息的队列，由VideoEnc线程Push Read Pop

    vector<Config> mv_configs;
    std::function<void(StateMachine &)> mf_fun;
    // queue<string> mq_folder; //输入图片文件夹路径
    shared_ptr<ThreadSafeQueue<string>> mq_folder; //输入图片文件夹路径
    string m_topic;                                // 当前正在处理的topic

    vector<unique_ptr<ThreadBase>>
        mvt_VideoEnc;                                    ///所有的视频线程
    vector<unique_ptr<ThreadBase>> mvt_frameEnc;         ///所有的图片转换线程
    vector<unique_ptr<ReadPictureFromFile>> mt_produces; ///所有的读取线程 未来多线程读取可能有用
    unique_ptr<ReadPictureFromFile> mt_produce;          ///读取线程

    double m_gmemory_allocate_costs = 0.0;
    double m_gmemory_free_costs = 0.0;

    /**@brief 管理运行中的线程\n
         * @details
         * 如果读取完成就sleep Frame线程，然后等待视频线程
         * 如果视频线程结束，就重置Frame线程。当三个线程均结束后，就切换到 StateMachine::init_task
*/
    void manage(void)
    {
        int q1_cnt = mq_jpg->Size();
        int q2_cnt = mq_yuv->Size();
        // int q3_cnt = mq_empty_box->Size();
        // char asd = mq_yuv->State;
        // printf("Queue Size(%d,%d,%d,%d)\n", q1_cnt, q2_cnt, q3_cnt, asd);
        size_t cnt = 0;
        // wait read thread finish
        for (auto &pthd : mt_produces)
        {
            if (pthd->State != ThreadBase::FINISH)
            {
                this_thread::sleep_for(chrono::milliseconds(100));
                return;
            }
        }
        if (q1_cnt == 0) //q1_cnt = 0 means Frame thread finished?
        {
            mq_jpg->State = ImageSyncQueue<std::shared_ptr<ImageStr>>::FINISH;
            while (true)
            {
                // 等待frameEnc 完成？？
                q2_cnt = mq_yuv->Size();
                // printf("Queue Size(%d,%d,%d,%d)\n", q1_cnt, q2_cnt, q3_cnt, asd);
                if (q2_cnt == 0)
                {
                    mq_yuv->State = ImageSyncQueue<std::shared_ptr<ImageStr>>::FINISH;
                    break;
                }
                this_thread::sleep_for(chrono::milliseconds(100));
            }
            while (true)
            {
                // 等待Video_enc 完成？？
                cnt = 0;
                // printf("Queue Size(%d,%d,%d,%d)\n", q1_cnt, q2_cnt, q3_cnt, asd);
                for (auto &pthd : mvt_VideoEnc)
                {
                    if (pthd->State == ThreadBase::STATE::WAIT)
                        cnt++;
                }
                if (cnt == mvt_VideoEnc.size())
                {
                    for (auto &pthd : mvt_frameEnc)
                    {
                        pthd->State = ThreadBase::FINISH;
                    }
                    // cout << "pthd->State = ThreadBase::FINISH;" << endl;
                    mf_fun = &StateMachine::init_task;
                    return;
                }
                this_thread::sleep_for(chrono::milliseconds(100));
            }
        }
        this_thread::sleep_for(chrono::milliseconds(100));
    }
    /**@brief 状态机开始 重置线程状态\n
         * @details
         * 释放环形队列内存
         * 检查任务是否完成，
         * 读取图片宽高，申请内存空间
         * 重置线程状态，开始执行
         * 切换到 StateMachine::init_task
*/
    void init_task()
    {
        if (mq_folder->size() == 0)
        {
            // mt_produce->State = ThreadBase::EXIT;
            for (auto &pthd : mt_produces)
            {
                pthd->State = ThreadBase::EXIT;
            }
            for (auto &pthd : mvt_frameEnc)
            {
                pthd->State = ThreadBase::EXIT;
            }
            for (auto &pthd : mvt_VideoEnc)
            {
                pthd->State = ThreadBase::EXIT;
            }
            mf_fun = &StateMachine::finish;
            return;
        }

        string pic_path;
        if (get_one_file(mq_folder->front(), pic_path) == -1) // 从图片目录读取到第一个jpg文件,并获取当前topicname
        {
            mq_folder->pop();
            return;
        }
        get_topic(mq_folder->front(), m_topic);
        size_t len = 0;
        uint8_t *pbuf = nullptr;

        fstream infile(pic_path, ios::in | ios::binary);
        if (!infile)
        {
            cerr << "inputFileName:" << pic_path << endl;
            assert(0);
        }

        // 获取jpg图片的大小信息，并将其读入内存
        infile.seekg(0, ios::end);
        len = infile.tellg();
        pbuf = new uint8_t[len];
        infile.seekg(0, ios::beg);
        infile.read(reinterpret_cast<char *>(pbuf), len);
        infile.close();

        struct gpujpeg_image_parameters param_image;
        int segment_count;
        cout << pic_path << "-" << len << endl;
        gpujpeg_reader_get_image_info(pbuf, len, &param_image, &segment_count, 0); // 从JPEG图片中获取图片的相关信息？
        // temp_timer.Start();
        // 根据公式给所有的YUV图像分配显存
        for (int i = mq_empty_box->Size(); i < mv_configs[0].cycle_que_len; i++)
        {
            std::shared_ptr<ImageStr> pimg = ImageStr::Create(10); // 10是什么含义？？？？？10个一组处理
            // TODO CPU MEMORY
            for (int j = 0; j < 10; j++)
            {
                // cudaError_t status = cudaMalloc((void **)&pimg->b[j].d_pImg, 1920 * 1080 * 1.5 + 10); //1382400
                cudaError_t status = cudaMalloc((void **)&pimg->b[j].d_pImg, mv_configs[0].width * mv_configs[0].height * 1.5 + 10); //1382400); // YUV420格式大小的计算公式，为啥加10？
                if (status != cudaSuccess)
                    printf("Error allocating pinned host memoryn");
            }
            mq_empty_box->Push(pimg);
        }
        // m_gmemory_allocate_costs += temp_timer.ElapsedSeconds();
        // set the queue state to RUNNING so the queue can be tryPop
        mq_empty_box->State = ImageSyncQueue<std::shared_ptr<ImageStr>>::RUNNING;
        mq_jpg->State = ImageSyncQueue<std::shared_ptr<ImageStr>>::RUNNING;
        mq_yuv->State = ImageSyncQueue<std::shared_ptr<ImageStr>>::RUNNING;

        for (auto &pthd : mt_produces)
        {
            pthd->folder_name = mq_folder->front();
            pthd->State = ThreadBase::STATE::INIT;
        }
        // mt_produce->State = ThreadBase::INIT; // read thread ready to work
        mq_folder->pop();
        for (auto &pthd : mvt_frameEnc)
        { // jpg2yuv thread ready to work
            pthd->State = ThreadBase::STATE::INIT;
        }
        for (auto &pthd : mvt_VideoEnc)
        { // yuv2hevc thread ready to work
            pthd->topic = m_topic;
            pthd->State = ThreadBase::STATE::RUNNING;
            // pthd->State = ThreadBase::STATE::EXIT;
        }
        // 等待队列中有数据，才能正式开始任务，切换至manage状态
        while (mq_jpg->Pop_cnt() == 0)
        {
            LOG(INFO) << "manage state";
            this_thread::sleep_for(chrono::milliseconds(100));
        }
        mf_fun = &StateMachine::manage;
    }
    /**@brief 状态机开始 根据配置文件开启各线程，然后切换到 StateMachine::init_task
*/
    void start(void)
    {
        for (int i = 0; i < mv_configs[0].img_thread_num; i++)
        {
            mvt_frameEnc.emplace_back(std::make_unique<JpgDec>(0, mq_jpg, mq_yuv));
        }

        // mt_produce = std::make_unique<ReadPictureFromFile>("path", mq_empty_box, mq_jpg); // 此时读线程开始
        for (int i = 0; i < mv_configs[0].read_thread; i++)
        {
            mt_produces.emplace_back(std::make_unique<ReadPictureFromFile>("path", 0, 1, mq_empty_box, mq_jpg)); // 此时读线程开始
        }

        for (int i = 0; i < mv_configs[0].video.thread_num; i++)
        {
            mvt_VideoEnc.emplace_back(std::move(std::make_unique<VideoEnc>(mv_configs[0], mq_yuv, mq_empty_box)));
        }
        mf_fun = &StateMachine::init_task;
    }

    /**@brief 状态机构造函数 开启定时器，读取配置，然后切换到 StateMachine::start
*/
    StateMachine(shared_ptr<ThreadSafeQueue<string>> folders) : mq_folder(folders)
    {
        m_start_clock = clock();
        m_timer.Start();
        read_config(mv_configs);
        LOG(INFO) << "start";
        mf_fun = &StateMachine::start;
    }
    Timer m_timer;
    clock_t m_start_clock = 0;
    clock_t m_end_clock = 0;
    bool isExit = false;

    void finish(void)
    {
        double frame_enc_costs = 0.0, video_enc_costs = 0.0;

        // mt_produce->join();
        for (auto &pthd : mt_produces)
        {
            pthd->join();
        }
        for (auto &pthd : mvt_frameEnc)
        {
            frame_enc_costs += pthd->Handle_time;
            pthd->join();
        }
        for (auto &pthd : mvt_VideoEnc)
        {
            video_enc_costs += pthd->Handle_time;
            pthd->join();
        }
        isExit = true;
        m_end_clock = clock();
        double real_time = m_timer.ElapsedSeconds();
        double use_cpu_time = (double)(m_end_clock - m_start_clock) / CLOCKS_PER_SEC; /*#define CLOCKS_PER_SEC ((clock_t)1000)*/

        cout.setf(ios::fixed);
        cout << "\tExecute Time:" << real_time << setprecision(3) << " 秒,"
             << "\tCPU Avarage Utilization Rate:(" << (int)(use_cpu_time / real_time * 100) << " %/ 1200 %"
             << "\tFrame encode costs:" << frame_enc_costs << " s"
             << "\tVideo encode costs:" << video_enc_costs << " s" << endl;
        //  << "\tMemory release costs:" << m_gmemory_free_costs << " s"
        //  << "\tMemory allocate costs:" << m_gmemory_allocate_costs << " s" << endl;
        // printf("Frame encode costs: %lf\n", frame_enc_costs);
        // printf("\t文件:%ld MiB,帧数:%d\n", 0 >> 20, 0);
    }
    void idle(void)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
};

/**@brief 通过命令行进入该函数 压缩指定目录的jpg文件
* @param[in]  argc              
* @param[in]  **argv                
* @return  0
*/
int compresse_dir(int argc, char **argv)
{
    vector<Config> configs;
    read_config(configs);
    auto folders = make_shared<ThreadSafeQueue<string>>();
    for (auto &it : configs[0].image.input_file_names)
    {
        folders->push(it);
    }
    vector<StateMachine> state_machines;
    for (int i = 0; i < configs[0].compress_pipe_num; i++)
    {
        state_machines.emplace_back(folders);
    }
    while (true)
    {
        bool is_finished = true;
        for (int i = 0; i < configs[0].compress_pipe_num; i++)
        {
            if (!state_machines[i].isExit)
            {
                state_machines[i].mf_fun(state_machines[i]);
                is_finished = false;
            }
        }
        if (is_finished)
            return 0;
    }
    return 0;
}

/**@brief 通过命令行进入该函数 执行bag图片压缩操作
* @param[in]  argc              
* @param[in]  **argv                
* @return  0
*/
int compresse_bag(int argc, char **argv)
{
    vector<Config> configs;
    read_config(configs);
    auto folders = make_shared<ThreadSafeQueue<string>>();
    get_bag_jpg_folders(folders, configs[0].local_bag);
    vector<StateMachine> state_machines;
    for (int i = 0; i < configs[0].compress_pipe_num; i++)
    {
        state_machines.emplace_back(folders);
    }
    while (true)
    {
        bool is_finished = true;
        for (int i = 0; i < configs[0].compress_pipe_num; i++)
        {
            if (!state_machines[i].isExit)
            {
                state_machines[i].mf_fun(state_machines[i]);
                is_finished = false;
            }
        }
        if (is_finished)
            return 0;
    }
    return 0;
}

/**@brief 通过命令行进入该函数 将bag中图片下载到指定文件夹下
* @param[in]  argc              
* @param[in]  **argv                
* @return  0
*/
int data_fun(int argc, char **argv)
{
    vector<Config> mv_configs;
    read_config(mv_configs);
    map<string, bool> Processed;
    vector<SavePictureFromDataSet *> pths;
    for (auto &it : mv_configs[0].topics)
    {
        if (it != "")
            pths.emplace_back(new SavePictureFromDataSet(mv_configs[0], it, nullptr, nullptr));
    }

    for (auto &pth : pths)
    {
        pth->Run();
    }

    for (auto &pth : pths)
    {
        pth->join();
    }

    return 0;
}