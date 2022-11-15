#pragma once
#include <iostream>
// #include "yaml-cpp/yaml.h"
#include <fstream>
// #include "../tool/json/json.h"
#include "json.h"

using namespace std;
#define WIDTH 20
#define Print std::cout << setw(WIDTH)

typedef struct
{
    int width;
    int height;
    vector<string> input_file_names;

    string input_file_name;  // input_path
    string output_file_name; // out_path

    void print()
    {
        cout.setf(ios::left); //设置对齐方式为left
        Print << "input_file_name:" << input_file_name << std::endl;
        Print << "output_file_name:" << output_file_name << std::endl;
        Print << "width,height:" << width << "," << height << std::endl;
    }

} image_para;
typedef struct
{
    int fps;
    string codec;
    int thread_num;
    void print()
    {
        cout.setf(ios::left); //设置对齐方式为left
        Print << "fps:" << fps << std::endl;
        Print << "codec:" << codec << std::endl;
    }
} video_para;
typedef struct
{

    int gpuid;
    int height;
    int width;
    int img_thread_num;
    int cycle_que_len;
    int read_thread;
    int compress_pipe_num;
    int dec_pool_size;
    string ds_bag;
    string ds_path;
    string local_bag;
    vector<string> topics;

    image_para image;
    video_para video;
    void print()
    {
        cout.setf(ios::left); //设置对齐方式为left
        Print << "gpuid:" << gpuid << std::endl;
        Print << "img_thread_num:" << img_thread_num << std::endl;
        Print << "ds_bag:" << ds_bag << std::endl;

        image.print();
        video.print();
    }

} Config;

void read_config(vector<Config> &configs)
{
    Json::Value root;
    std::ifstream config_doc("../src/config.json", std::ifstream::binary);
    // std::ifstream config_doc("/home/tusimple/jianyu.liu/image-video-codec-pipeline-feature-test/src/config.json", std::ifstream::binary);
    if (!config_doc)
    {
        std::cout << "no such json file,please check the file name!\n";
        exit(0);
    }
    try
    {
        config_doc >> root;
    }
    catch (const std::exception &e)
    {
        std::cerr << "JSON err,please check config.json file " << '\n';
        exit(0);
    }

    Json::CharReaderBuilder rbuilder;
    std::string errs;
    Json::parseFromStream(rbuilder, config_doc, &root, &errs);

    for (auto &rt : root)
    {
        Config config;

        config.gpuid = rt["gpu"].asInt();
        config.width = rt["jpg_width"].asInt();
        config.height = rt["jpg_height"].asInt();
        config.img_thread_num = rt["pic_thread"].asInt();
        config.image.input_file_name = rt["input_path"].asString();
        for (unsigned int i = 0; i < rt["pic_path"].size(); i++)
        {
            config.image.input_file_names.push_back(rt["pic_path"][i]["url"].asString());
        }
        for (unsigned int i = 0; i < rt["topics"].size(); i++)
        {
            config.topics.push_back(rt["topics"][i]["url"].asString());
        }
        config.image.output_file_name = rt["out_path"].asString();
        config.ds_bag = rt["ds_bag"].asString();
        config.ds_path = rt["ds_path"].asString();
        config.local_bag = rt["local_bag"].asString();
        config.video.thread_num = rt["video_thread"].asInt();
        config.read_thread = rt["read_thread"].asInt();
        config.video.fps = rt["fps"].asInt();
        config.cycle_que_len = rt["cycle_que_len"].asInt();
        config.compress_pipe_num = rt["compress_pipe_num"].asInt();
        config.dec_pool_size = rt["dec_pool_size"].asInt();

        configs.push_back(config);
    }
    config_doc.close();
}
