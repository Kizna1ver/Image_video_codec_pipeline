#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <iostream>

#include <sys/types.h>
#include <dirent.h>
#include <stdio.h>
#include <errno.h>

#include "ThreadSaveQueue.hpp"

#define RESET "\033[0m"
#define BLACK "\033[30m"  /* Black */
#define RED "\033[31m"    /* Red */
#define GREEN "\033[32m"  /* Green */
#define YELLOW "\033[33m" /* Yellow */
#define BLUE "\033[34m"   /* Blue */
#define PURPLE "\033[35m" /* Purple */
#define CYAN "\033[36m"   /* Cyan */
#define WHITE "\033[37m"  /* White */

#define CHECK_LOG(CALL, INFO)                                                                                   \
    if (CALL)                                                                                                   \
    {                                                                                                           \
        std::cout << "\033[31m[ERROR]\033[37m " << INFO << "  " << __LINE__ << " in file " << __FILE__ << endl; \
        return EXIT_FAILURE;                                                                                    \
    }
// #define INFO_LOG std::cout << "\033[32m[INFO]\033[37m "
using namespace std;

class media_info
{
    struct _data
    {
        int width;
        int height;
        int pic_number;
        char filename[64];
    };

public:
    struct _data data;
    std::vector<uint64_t> timestamps;
    std::vector<uint64_t> frame_base;

    void save(string savepath, string savename)
    {
        ofstream outFile(savepath + savename + ".dat", ios::out | ios::binary);
        data.pic_number = timestamps.size();
        sprintf(data.filename, "%s", savename.c_str());
        outFile.write(reinterpret_cast<char *>(&data), sizeof(struct _data));
        outFile.write(reinterpret_cast<char *>(timestamps.data()), timestamps.size() * sizeof(uint64_t));
        outFile.write(reinterpret_cast<char *>(frame_base.data()), frame_base.size() * sizeof(uint64_t));
        outFile.close();
        // cout << savename << "number:" << timestamps.size() << endl;
    }

    int read(string savename)
    {
        fstream file(savename, ios::in | ios::binary);
        if (!file)
        {
            cout << "Error opening file." << savename << endl;
            return 0;
        }
        file.read(reinterpret_cast<char *>(&data), sizeof(struct _data));
        // file.seekg(sizeof(struct _data),ios::beg);
        timestamps.resize(data.pic_number);
        frame_base.resize(data.pic_number);
        file.read(reinterpret_cast<char *>(timestamps.data()), data.pic_number * sizeof(uint64_t));
        // file.seekg(sizeof(struct _data)+data.pic_number * sizeof(uint64_t));
        file.read(reinterpret_cast<char *>(frame_base.data()), data.pic_number * sizeof(uint64_t));
        // std::copy(pdata, pdata + data.pic_number, ti8mestamps.begin());

        // printf("read:%s,%s,frame_base %ld\n", savename.c_str(), data.filename, frame_base.size());
        file.close();
        return 0;
    }
};

int scan_files(vector<string> &fileList, string inputDirectory);
// int get_one_file(string inputDirectory, string &outputFileName);
void get_topic(string input, string &topic);
int get_bag_jpg_folders(shared_ptr<ThreadSafeQueue<string>> folders, string local_bag_path);
int get_jpg_folders(shared_ptr<ThreadSafeQueue<string>> folders, string local_bag_path);

// get xxx.dat file lists
int scan_files(vector<string> &fileList, string input_dir)
{
    DIR *p_dir;
    const char *str = input_dir.c_str();

    p_dir = opendir(str);
    if (p_dir == NULL)
    {
        cout << "can't open :" << input_dir << endl;
    }

    struct dirent *p_dirent;

    while (true)
    {
        p_dirent = readdir(p_dir);
        if (p_dirent == NULL)
            break;
        string tmpFileName = p_dirent->d_name;

        if (tmpFileName == "." || tmpFileName == "..")
        {
            continue;
        }
        else
        {
            string::size_type offisetpre = tmpFileName.rfind(".dat", tmpFileName.npos);
            if (offisetpre != string::npos)
            {
                // cout << "FIND:" << tmpFileName << ":" << (offisetpre == string::npos) << endl;
                fileList.push_back(tmpFileName);
            }
        }
    }
    closedir(p_dir);
    return fileList.size();
}

void get_topic(string input, string &topic)
{
    int idx = input.size() - 2;
    while (input[idx] != '/')
    {
        idx--;
    }
    topic = input.substr(0, idx);
    idx = topic.size();
    while (topic[idx] != '/')
    {
        idx--;
    }
    topic = topic.substr(idx);
    cout << "topic name:" << topic << endl;
}

int get_one_file(string input_dir, string &outputFileName)
{
    DIR *p_dir;
    const char *str = input_dir.c_str();

    p_dir = opendir(str);
    if (p_dir == NULL)
    {
        cout << RED << "can't open :" << input_dir << WHITE << endl;
        return -1;
    }
    struct dirent *p_dirent;
    while (true)
    {
        p_dirent = readdir(p_dir); // 依次读取目录下的文件
        if (p_dirent == NULL)
            break;
        string tmpFileName = p_dirent->d_name;

        if (tmpFileName == "." || tmpFileName == "..")
        {
            continue;
        }
        else
        {
            string::size_type offisetpre = tmpFileName.rfind(".jpg", tmpFileName.npos);
            if (offisetpre != string::npos)
            {
                // cout << "FIND:" << tmpFileName << ":" << (offisetpre == string::npos) << endl;
                outputFileName = input_dir + tmpFileName;
                closedir(p_dir);
                return 0;
            }
        }
    }
    cout << "folder " << input_dir << " contained nothing,it has been skipped" << endl;
    closedir(p_dir);
    return -1;
}

int get_bag_jpg_folders(shared_ptr<ThreadSafeQueue<string>> folders, string local_bag_path)
{
    DIR *p_dir;
    const char *str = local_bag_path.c_str();

    p_dir = opendir(str);
    if (p_dir == NULL)
    {
        cout << "can't open :" << local_bag_path << endl;
    }

    struct dirent *p_dirent;

    while (true)
    {
        p_dirent = readdir(p_dir);
        if (p_dirent == NULL)
            break;
        string tmpFileName = p_dirent->d_name;

        if (tmpFileName == "." || tmpFileName == "..")
            continue;
        else
        {
            string topic_folder = local_bag_path + tmpFileName + "/";
            get_jpg_folders(folders, topic_folder);
        }
    }
    closedir(p_dir);
    return folders->size();
}

int get_jpg_folders(shared_ptr<ThreadSafeQueue<string>> folders, string local_bag_path)
{
    DIR *p_dir;
    const char *str = local_bag_path.c_str();

    p_dir = opendir(str);
    if (p_dir == NULL)
    {
        cout << "can't open :" << local_bag_path << endl;
    }

    struct dirent *p_dirent;

    while (true)
    {
        p_dirent = readdir(p_dir);
        if (p_dirent == NULL)
            break;
        string tmpFileName = p_dirent->d_name;

        if (tmpFileName == "." || tmpFileName == "..")
            continue;
        else
        {
            string jpg_folder = local_bag_path + tmpFileName + "/";
            folders->push(jpg_folder);
        }
    }
    closedir(p_dir);
    return folders->size();
}

inline bool m_check(int e, int iLine, const char *szFile)
{
    if (e < 0)
    {
        std::cout << "General error " << e << " at line " << iLine << " in file " << szFile;
        return false;
    }
    return true;
}
inline bool checkval(int e, int val, int iLine, const char *szFile)
{
    if (e == !val)
    {
        std::cout << "General error " << e << " at line " << iLine << " in file " << szFile;
        return false;
    }
    return true;
}
#define CHECK(call) m_check(call, __LINE__, __FILE__)
#define CHECK_EQ(call, val) checkval(call, val, __LINE__, __FILE__)
