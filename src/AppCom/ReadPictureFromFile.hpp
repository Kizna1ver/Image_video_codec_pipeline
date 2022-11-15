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
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thread>

#include "../tool/ImageSyncQueue.hpp"

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

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

class ReadPictureFromFile : public ThreadBase
{
public:
    string folder_name;
    int thd_idx; // 弃用
    int thd_num; // 弃用
    ImageStr src_pic_info;
    atomic_bool mutex_{false};
    int gpuid = 0;
    shared_ptr<ImageSyncQueue<string>> mpq_pic_names;

    ReadPictureFromFile(string input_file_name, int thd_idx, int thd_num, const QS_queue &Poutput, const QS_queue &Poutput2) : ThreadBase(Poutput, Poutput2), folder_name(input_file_name), thd_idx(thd_idx), thd_num(thd_num)
    {
        gpuid = 0;
        Run();
    }
    ~ReadPictureFromFile(){};
    static int scan_files(vector<string> &file_list, string input_dir) // get all files path in input directory
    {
        DIR *p_dir;
        const char *str = input_dir.c_str();

        p_dir = opendir(str);
        if (p_dir == NULL)
        {
            cout << "can't open :" << input_dir << endl;
            return -1;
        }

        struct dirent *p_dirent;

        while (true)
        {
            p_dirent = readdir(p_dir);
            if (p_dirent == NULL)
                break;
            string temp_file_name = p_dirent->d_name;

            if (temp_file_name == "." || temp_file_name == "..")
            {
                continue;
            }
            else
            {
                // cout << "FIND:" << input_dir+temp_file_name << endl;
                file_list.push_back(input_dir + temp_file_name);
            }
        }
        closedir(p_dir);
        return file_list.size();
    }

    void get_ts(uint64_t &ts, const string file_name)
    {
        int idx = file_name.size() - 1;
        while (file_name[idx] != '/')
        {
            idx--;
        }
        string ts_str = file_name.substr(idx + 1);
        ts = stoull(ts_str);
    }

    int image_num;

    void run()
    {
        vector<string> file_list;
        size_t k = thd_idx;
        while (true)
        {
            int fg = 0xff;
            bool ret;
            switch (State)
            {
            case ThreadBase::WAIT:
                std::this_thread::yield(); //   this_thread::sleep_for(chrono::milliseconds(10));
                continue;
            case ThreadBase::CREATE:
                LOG(INFO) << "  Read thread is ready!";
                State = ThreadBase::WAIT;
                continue;
            case ThreadBase::FINISH:
                k = thd_idx;               // reset k
                std::this_thread::yield(); //  this_thread::sleep_for(chrono::milliseconds(10));
                continue;
            case ThreadBase::EXIT:
                LOG(INFO) << "  Read end and exit!";
                return;
            case ThreadBase::INIT:

                file_list.clear();
                scan_files(file_list, folder_name);
                sort(file_list.begin(), file_list.end());
                State = ThreadBase::RUNNING;
                k = thd_idx; // reset k
                LOG(INFO) << "  Total image count:" << file_list.size() << endl;
                break;
            case ThreadBase::RUNNING:
            { // Read jpg file‘s info to memory
                if (k >= file_list.size())
                {
                    State = ThreadBase::FINISH;
                    LOG(INFO) << "Read Thread" << thd_idx << " FINISH  " << file_list.size() << "\tTime costs: " << Handle_time << " s";
                    // output->State = ImageSyncQueue<std::shared_ptr<ImageStr>>::FINISH;
                    this_thread::sleep_for(chrono::milliseconds(1000));
                    continue;
                }
                std::shared_ptr<ImageStr> pImageStr; //=   ImageStr::Create(10);
                ret = input->tryPop(pImageStr, fg);
                // cout << "ret:" << ret << "fg: " << fg;
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
                pImageStr->message_len = 0;
                Timer temp_timer;
                temp_timer.Start();
                for (size_t j = 0; j < 10 && k < file_list.size(); j++, k += thd_num)
                {

                    fstream infile(file_list[k], ios::in | ios::binary);
                    if (!infile)
                    {
                        cerr << "inputFileName:" << file_list[k] << endl;
                        assert(0);
                    }

                    infile.seekg(0, ios::end);
                    pImageStr->b[j].size = infile.tellg();
                    pImageStr->b[j].pImg = new uint8_t[pImageStr->b[j].size];
                    infile.seekg(0, ios::beg);
                    infile.read(reinterpret_cast<char *>(pImageStr->b[j].pImg), pImageStr->b[j].size);
                    infile.close();
                    // pImageStr->b[j].pImg = loadFile(pImageStr->b[j].size, file_list[k]);
                    // pImageStr->b[j].timestamp = Timer::GetUTC();
                    get_ts(pImageStr->b[j].timestamp, file_list[k]);
                    pImageStr->b[j].isvalue = true;
                    pImageStr->message_len++;
                }
                Handle_time += temp_timer.ElapsedSeconds();
                // printf("Read JPG costs: %lf\n", temp_timer.ElapsedSeconds());
                // if ((k - thd_idx) % 1000 == 0)
                //     printf("[LOG ] Read Thread:count=%ld\n", k);
                pImageStr->_id = k;
                output->Push(pImageStr);
                break;
            }
            default:
                break;
            }
        }
    }
};
