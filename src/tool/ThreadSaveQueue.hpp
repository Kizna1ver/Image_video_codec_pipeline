#include <iostream>
#include <string>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <memory>

template <class T, class Container = std::queue<T>>
class ThreadSafeQueue
{
public:
    ThreadSafeQueue() = default;

    void push(const T &element)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(element);
    }

    T pop()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        T t_ptr = queue_.front();
        queue_.pop();

        return t_ptr;
    }

    T front()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.front();
    }

    bool is_empty() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    int size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    ThreadSafeQueue(const ThreadSafeQueue &) = delete;
    ThreadSafeQueue &operator=(const ThreadSafeQueue &) = delete;
    ThreadSafeQueue(ThreadSafeQueue &&) = delete;
    ThreadSafeQueue &operator=(ThreadSafeQueue &&) = delete;

private:
    Container queue_;

    // std::condition_variable not_empty_cv_;
    mutable std::mutex mutex_;
};