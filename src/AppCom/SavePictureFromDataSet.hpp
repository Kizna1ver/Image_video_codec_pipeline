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
#include <regex>

class SavePictureFromDataSet : public ThreadBase
{
    struct ImageSize
    {
        int width_;  ///< image widh
        int height_; ///< image height
    };

    struct gpujpeg_decoder *add_decoders()
    {
        struct gpujpeg_decoder *decoder = gpujpeg_decoder_create(stream);
        struct gpujpeg_parameters param;

        struct gpujpeg_image_parameters param_image;

        gpujpeg_set_default_parameters(&param); // show debug info
        gpujpeg_image_set_default_parameters(&param_image);

        ck(gpujpeg_decoder_init(decoder, &param, &param_image));
        gpujpeg_decoder_set_output_format(decoder, GPUJPEG_YUV, GPUJPEG_420_U8_P0P1P2); // key para
        gpujpeg_decoder_output_set_cuda_buffer(&decoder_output);
        // cout<<"pic quaty:"<<param.quality<<endl;
        return decoder;
    }

public:
    ImageStr src_pic_info;

    int gpuid = 0;
    Config param;
    vector<string> *pProcessed;
    const QS_queue *pPoutput2;
    std::string topic_;
    SavePictureFromDataSet(Config &p_param, string &topic_name, const QS_queue &Poutput, const QS_queue &Poutput2) : ThreadBase(Poutput, Poutput)
    {
        param = p_param;
        gpuid = param.gpuid;
        topic_ = topic_name; //"/camera17/image_color/compressed"; // 2K + 1K images

        folder_name = ""; //param.image.input_file_name; // "data4/nasa_1280/";
        State = ThreadBase::STATE::CREATE;
        pProcessed = new vector<string>();
        pPoutput2 = &Poutput2;
    }
    ~SavePictureFromDataSet(){};

private:
    string folder_name;

    cudaStream_t stream;
    struct gpujpeg_decoder_output decoder_output;

    void put_queue()
    {
        static int cnt = 0;
        if (cnt == 0)
        {
        }
    }
    void run()
    {
        // TODO
        // we remove all of the code about dataset
    }
};
