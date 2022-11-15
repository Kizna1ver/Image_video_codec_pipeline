
#ifndef ____THREADING__H__
#define ____THREADING__H__

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "ImageSyncQueue.hpp"
#include <queue>
#include "stdio.h"
#include <iostream>
#include <unistd.h>

class ThreadBase
{

public:
    std::QS_queue input;
    std::QS_queue output;
    uint64_t hd_cnt = 0; // what it means?the num of frame has been convert?
    static atomic<int> third_cnt;
    vector<string> pros;
    float Handle_time = 0; // never used
    atomic_char State;
    atomic<long int> frqe{0};
    atomic_char sign{0};
    string topic; // current topic

    ThreadBase(const QS_queue &Pinput, const QS_queue &Poutput)
    {
        output = Poutput;
        input = Pinput;
        State = ThreadBase::CREATE;
    }
    // ThreadBase(const shared_ptr<ImageSyncQueue<ImageStr>> &Pinput, const shared_ptr<ImageSyncQueue<ImageStr>> &Poutput)
    // {
    // }

    ThreadBase()
    {
    }
    void Run()
    {
        pthd = new thread(&ThreadBase::run, this);
    }
    uint64_t join()
    {
        if (pthd->joinable())
        {
            pthd->join();
        }
        return hd_cnt;
    }
    enum STATE
    {
        RUNNING,
        FINISH,
        SLEEP,
        CREATE,
        EXIT,
        INIT,
        WAIT
    };

private:
    thread *pthd;
    virtual void run() = 0;
};
atomic<int> ThreadBase::third_cnt{0};

#endif