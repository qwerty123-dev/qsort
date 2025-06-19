#include <vector>
#include <future>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <memory>
#include <iostream>
#include <random>
#include <chrono>

// ======== Thread Pool with Work Queue ========
class ThreadPool {
public:
    ThreadPool(size_t num_threads) {
        for(size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this]() {
                while (true) {
                    Task task;
                    {
                        std::unique_lock lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker: workers)
            worker.join();
    }

    template<class F>
    void push_task(F&& f) {
        {
            std::unique_lock lock(queue_mutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

private:
    using Task = std::function<void()>;

    std::vector<std::thread> workers;
    std::queue<Task> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop = false;
};

// ======== QuickSort with async task tracking ========
void quicksort(std::vector<int>& arr, int left, int right,
               ThreadPool& pool,
               std::shared_ptr<std::atomic<int>> counter,
               std::shared_ptr<std::promise<void>> done)
{
    if (left >= right) {
        if (--(*counter) == 0) {
            done->set_value();
        }
        return;
    }

    int pivot = arr[(left + right) / 2];
    int l = left, r = right;

    while (l <= r) {
        while (arr[l] < pivot) l++;
        while (arr[r] > pivot) r--;
        if (l <= r) std::swap(arr[l++], arr[r--]);
    }

    if (r - left > 100000) {
        counter->fetch_add(1);
        pool.push_task([=, &arr, &pool]() {
            quicksort(arr, left, r, pool, counter, done);
        });
    } else {
        quicksort(arr, left, r, pool, counter, done);
    }

    if (right - l > 100000) {
        counter->fetch_add(1);
        pool.push_task([=, &arr, &pool]() {
            quicksort(arr, l, right, pool, counter, done);
        });
    } else {
        quicksort(arr, l, right, pool, counter, done);
    }

    if (--(*counter) == 0) {
        done->set_value();
    }
}

// ======== Entry Point ========
void parallel_quicksort(std::vector<int>& arr, ThreadPool& pool) {
    auto counter = std::make_shared<std::atomic<int>>(1);
    auto done = std::make_shared<std::promise<void>>();
    pool.push_task([&arr, &pool, counter, done]() {
        quicksort(arr, 0, arr.size() - 1, pool, counter, done);
    });
    done->get_future().wait(); 
}

int main() {
    const int N = 10'000'000;
    std::vector<int> data(N);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 1'000'000);
    for (auto& x : data) x = dist(rng);

    ThreadPool pool(std::thread::hardware_concurrency());

    auto start = std::chrono::high_resolution_clock::now();
    parallel_quicksort(data, pool);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Sorted " << N << " elements in " << elapsed.count() << " seconds.\n";

    return 0;
}
