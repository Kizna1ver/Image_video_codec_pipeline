#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>

using namespace std;
#define WIDTH 20
#define Print std::cout << setw(WIDTH)
#define CAPACITY 1000

struct img_info
{						  ///  @brief 保留单张图片信息的结构体
	size_t size = 0;	  ///< 图片大小
	uint8_t *pImg;		  ///< jpg图片在CPU内存中的地址
	uint8_t *d_pImg;	  ///< YUV图片在GPU内存中的地址
	uint64_t timestamp;	  ///< 时间辍
	bool isvalue = false; ///<该图片是否有效 已弃用
	string topic;
	unsigned int height = 0;
	unsigned int width = 0;
	/**@brief 创建一个该结构体的智能指针
* @param[out]  该结构体的智能指针                
* @return  shared_ptr<img_info>
*/
	static shared_ptr<img_info> Create()
	{
		shared_ptr<img_info> p = make_shared<img_info>();
		return p;
	}
};

// 包装img_info？
class ImageStr
{

public:
	typename std::shared_ptr<ImageStr> Ptr;
	struct img_info *b;
	uint8_t *pImg;
	uint8_t *d_pImg;
	int size;
	int d_size;
	int _id;

	uint64_t timestamp;
	bool valid_ = false;
	int message_len = 0;
	//   cv::Mat img;
	ImageStr(int _size)
	{
		pImg = nullptr;
		size = 0;
		timestamp = 0;

		message_len = 0;
		b = new img_info[_size];
	}
	ImageStr()
	{
		pImg = nullptr;
		size = 0;
		timestamp = 0;

		message_len = 0;
		b = new img_info[10];
	}
	~ImageStr()
	{
		// delete b;
	}
	void print()
	{
		cout.setf(ios::left); //设置对齐方式为left
		Print << "size:" << size << std::endl;
		// Print << "width,height:" << width << "," << height << std::endl;
		Print << "timestamp:" << timestamp << std::endl;
	}
	/**@brief 创建一个该结构体的智能指针
* @param[out]  该结构体的智能指针                
* @return  shared_ptr<img_info>
*/

	static std::shared_ptr<ImageStr> Create(int size)
	{
		return make_shared<ImageStr>(size);
	}
};

template <class T>
class ImageSyncQueue
{
public:
	enum
	{
		RUNNING,
		FINISH,
		SLEEP,
		CREATE,
		EXIT,
		INIT,
		EMPTY,
		OK,
	};
	atomic_char State; // 主状态机中改变state

protected:
	// Data
	std::queue<T> _queue;
	typename std::queue<T>::size_type _size_max;
	uint64_t pop_cnt = 0;
	// Thread gubbins
	std::mutex _mutex;
	std::condition_variable _fullQue; // 好像没用到？
	std::condition_variable _empty;
	// 原子操作
	std::atomic_bool _quit; //{ false };

public:
	std::atomic_bool _finished; // { false };
	typedef std::shared_ptr<ImageSyncQueue> Ptr;

	ImageSyncQueue(const size_t size_max)
	{
		_size_max = size_max;
		_quit = ATOMIC_VAR_INIT(false);
		_finished = ATOMIC_VAR_INIT(false);
		State = CREATE;
	}

	bool Push(T &data)
	{
		std::unique_lock<std::mutex> lock(_mutex);
		while (!_quit && !_finished)
		{
			// if (_queue.size() < _size_max)
			// {
			_queue.push(std::move(data));
			//_queue.push(data);
			if (_queue.size() == 1)
				_empty.notify_one();
			else
				_empty.notify_all();
			return true;
			// }
			// else
			// {
			// 	// wait的时候自动释放锁，如果wait到了会获取锁
			// 	_fullQue.wait(lock);
			// }
		}

		return false;
	}

	bool Pop(T &data)
	{
		std::unique_lock<std::mutex> lock(_mutex);
		while (!_quit)
		{
			if (!_queue.empty())
			{
				//data = std::move(_queue.front());
				data = _queue.front();
				_queue.pop();
				pop_cnt++;
				// _fullQue.notify_all();
				if (_queue.empty())
					_empty.notify_all();
				return true;
			}
			else if (_queue.empty() && _finished)
			{
				return false;
			}
			else
			{
				_empty.wait(lock);
			}
		}
		return false;
	}
	bool tryPop(T &data, int &flag)
	{
		std::unique_lock<std::mutex> lock(_mutex);
		while (!_quit)
		{
			if (!_queue.empty())
			{
				//data = std::move(_queue.front());
				data = _queue.front();
				_queue.pop();
				pop_cnt++;
				// _fullQue.notify_all();
				if (_queue.empty())
					_empty.notify_all();
				flag = State;
				return true;
			}
			else if (_queue.empty() && _finished)
			{
				return false;
			}
			else
			{
				// cout << "???" << endl;
				flag = State;
				return false;
			}
		}
		return false;
	}
	// The queue has finished accepting input
	void finished()
	{
		_finished = true;
		_empty.notify_all();
	}

	void quit()
	{
		_quit = true;
		_empty.notify_all();
		// _fullQue.notify_all();
	}
	void Wait()
	{
		std::unique_lock<std::mutex> lock(_mutex);
		while (!_queue.empty())
		{
			_empty.wait(lock);
		}
	}
	size_t Size()
	{
		std::unique_lock<std::mutex> lock(_mutex);
		return (_queue.size());
	}
	uint64_t Pop_cnt()
	{
		std::unique_lock<std::mutex> lock(_mutex);
		return pop_cnt;
	}
};

#define QS_queue shared_ptr<ImageSyncQueue<shared_ptr<ImageStr>>>
#define CreatQS_queue(val) std::make_shared<ImageSyncQueue<shared_ptr<ImageStr>>>(val)
