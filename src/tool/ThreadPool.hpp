#pragma once
#include <vector>
#include <queue>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <iostream>
#include <omp.h>
#include <chrono>
class ThreadPool
{
private:
    using Task = std::function<void()>; //using 相当于typedef
    std::vector<std::thread> m_threads;
    std::queue<Task> m_tasks; //任务队列
    std::mutex m_lock;
    std::condition_variable m_cvTask; //条件变量
    std::atomic<bool> m_stoped;       //是否关闭提交 std::atomic<T> 对于对象的操作都是原子的不用加锁
    std::atomic<int> m_idlThrNum;     //空闲线程数量
    //std::atomic<bool> m_stoped;
    std::atomic<int> m_taskNum;    // 当前任务队列中的任务数量
    std::atomic<int> m_maxTaskNum; //任务队列的最大任务数量
    std::thread *m_notifyThread;

public:
    ThreadPool(int size = 4, int maxTask = 5)
    {
        m_idlThrNum = size < 1 ? 1 : size; //如果传入的构造线程数量为1以下，那默认为1
        m_maxTaskNum = maxTask < 1 ? 1 : maxTask;
        m_taskNum.store(0);
        m_stoped.store(false);
        for (size = 0; size < m_idlThrNum.load(); ++size)
        {
            m_threads.emplace_back([this]
                                   {
                                       // 无限循环跑任务,没任务时就阻塞
                                       while (!this->m_stoped) //如果关闭为假，执行循环
                                       {
                                           Task task;
                                           {
                                               std::unique_lock<std::mutex> lock(this->m_lock);
                                               this->m_cvTask.wait(lock, [this]
                                                                   {
                                                                       //当停止为真或者任务队列不为空时条件变量触发，取消阻塞状态
                                                                       return this->m_stoped.load() || !this->m_tasks.empty();
                                                                   });
                                               //当条件变量触发前一直阻塞在这里
                                               if (this->m_stoped && this->m_tasks.empty())
                                                   return;                              //当触发线程池停止时，任务队列也为空，就结束线程
                                               task = std::move(this->m_tasks.front()); // 取一个 task
                                               this->m_tasks.pop();
                                           }
                                           m_idlThrNum--; //std::atomic<int> 是原子的，因此这里不用加锁
                                           task();
                                           cudaDeviceReset();
                                           //    std::cout << "Current task finished,wait next \n";
                                           m_idlThrNum++;
                                       }
                                   });
        }
        //单独的线程用于通知线程池执行任务
        m_notifyThread = new std::thread([this]
                                         {
                                             while (!this->m_stoped.load())
                                             {
                                                 if (m_idlThrNum.load() > 0 && m_taskNum.load() > 0)
                                                 {
                                                     //m_taskNum--;
                                                     int _trmp = m_taskNum.load() - 1;
                                                     m_taskNum.store(_trmp);
                                                     m_cvTask.notify_one(); // 唤醒一个线程执行
                                                     printf("---------------notify_one---%d---------- \n", _trmp);
                                                 }
                                                 //    int num=
                                                 this_thread::sleep_for(chrono::milliseconds(1));
                                                 //    printf("m_idlThrNum%d,%d\n", m_idlThrNum.load(),m_taskNum.load());
                                                 //    std::this_thread::yield();
                                             }
                                         }); //通知线程
        m_notifyThread->detach();
    }

    inline ~ThreadPool()
    {
        m_stoped.store(true);
        while (true)
        {
            m_cvTask.notify_all(); // 唤醒所有线程执行
            size_t cnt = 0;
            for (std::thread &thread : m_threads)
            {
                //thread.detach(); // 让线程“自生自灭”
                if (thread.joinable())
                {
                    thread.join(); // 等待任务结束， 前提：线程一定会执行完
                }
                else
                {
                    cnt++;
                }
            }
            if (cnt == m_threads.size())
            {
                break;
            }
        }
        std::cout << "OVER ";
    }

    template <class F, class... Args>
    auto commit(F &&f, Args &&...args) -> std::future<decltype(f(args...))>
    {

        if (m_stoped.load())
        {
            throw std::runtime_error("commit on ThreadPool is stopped.");
        }
        using RetType = decltype(f(args...)); //获取函数f的返回值类型
        auto task = std::make_shared<std::packaged_task<RetType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        //如果任务队列已达到最大容量，阻塞线程，等待历史任务处理
        while (m_taskNum.load() > m_maxTaskNum.load())
        {
            //std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            printf("asd\n");
        }
        std::future<RetType> future = task->get_future();
        {
            std::lock_guard<std::mutex> lock(m_lock);
            m_tasks.emplace(
                [task]()
                {
                    (*task)();
                });
        }
        std::cout << "commit one!current task num: " << m_taskNum << endl;
        m_taskNum++;
        //m_cvTask.notify_one(); // 唤醒一个线程执行
        return future;
    }
    //空闲线程数量
    int idlCount() const
    {
        return m_idlThrNum;
    }
};
