#pragma once

#include <chrono>
#include <string>
#include <sys/time.h>
// #define WIDTH 20
// #ifndef Print
// #define Print std::cout << std::setw(WIDTH)
// #endif

using namespace std::chrono;

class Timer
{
public:
  Timer() : started_(false), paused_(false) {}

  void Start()
  {
    started_ = true;
    paused_ = false;
    start_time_ = high_resolution_clock::now();
  }
  void Restart()
  {
    started_ = false;
    Start();
  }
  void Pause()
  {
    paused_ = true;
    pause_time_ = high_resolution_clock::now();
  }
  void Resume()
  {
    paused_ = false;
    start_time_ += high_resolution_clock::now() - pause_time_;
  }
  void Reset()
  {
    started_ = false;
    paused_ = false;
  }

  double ElapsedMicroSeconds() const
  {
    if (!started_)
    {
      return 0.0;
    }
    if (paused_)
    {
      return duration_cast<microseconds>(pause_time_ - start_time_).count();
    }
    else
    {
      return duration_cast<microseconds>(high_resolution_clock::now() -
                                         start_time_)
          .count();
    }
  }
  double ElapsedSeconds() const { return ElapsedMicroSeconds() / 1e6; }
  double ElapsedMinutes() const { return ElapsedSeconds() / 60; }
  double ElapsedHours() const { return ElapsedMinutes() / 60; }

  void PrintSeconds(const std::string &str) const
  {
    std::cout.setf(std::ios::left); //设置对齐方式为left
    Print << str << ElapsedSeconds() << "\n";
  }
  void PrintMinutes() const
  {
    printf("Elapsed time: %.3f [minutes]\n", ElapsedMinutes());
  }

  void PrintHours() const
  {
    printf("Elapsed time: %.3f [hours]\n", ElapsedHours());
  }

  static __time_t GetUTC()
  {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    // stringstream s;
    // s<<tv.tv_sec;
    // printf("second:%ld \n", tv.tv_sec);                                 //秒
    // printf("millisecond:%ld \n", tv.tv_sec * 1000 + tv.tv_usec / 1000); //毫秒
    // printf("microsecond:%ld \n", tv.tv_sec * 1000000 + tv.tv_usec);     //微秒
    return tv.tv_sec * 1000000 + tv.tv_usec;
  }
  static std::string GetUTCString()
  {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return std::to_string(tv.tv_sec * 1000000 + tv.tv_usec);
  }

private:
  bool started_;
  bool paused_;
  std::chrono::high_resolution_clock::time_point start_time_;
  std::chrono::high_resolution_clock::time_point pause_time_;
};
