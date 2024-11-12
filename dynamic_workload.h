#include <EGL/egl.h>
#include <GLES3/gl31.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <signal.h>
#include <random>
#include <map>
#include <yaml-cpp/yaml.h> // for yaml parameter.

class Workload {
 public:
  Workload();
  // Workload(int duration, int cpu, int gpu, bool random);
  Workload(int init_wait_time);

  ~Workload();

  void GPU_Worker();
  void CPU_Worker();

  int total_duration;
  float cpugpu_transition;
  int gpu_kernel_size;
  int cpu_cores;

  struct timespec start_time;
  bool ignition = false;
  bool cpu_ignition = false;
  bool gpu_ignition = false;
  bool gpu_kernel_done = false;
  bool terminate = false;
  std::mutex mtx;
  std::mutex cpu_mtx;
  std::mutex gpu_mtx;

  std::condition_variable cv;
  std::condition_variable cpu_cv;
  std::condition_variable gpu_cv;
  std::condition_variable gpu_end_cv;
  
  std::atomic_bool stop;
  std::atomic_bool cpu_stop;
  std::atomic_bool gpu_stop;
  std::atomic_bool cpu_worker_termination;
  std::atomic_bool gpu_worker_termination;


 private:
  // GPU workload pool
  std::vector<std::thread> gpu_workload_pool;

  // CPU workload pool
  std::vector<std::thread> cpu_workload_pool;

};  // class Workload