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
#include <iostream>
#include <cstdlib>
#include <signal.h>
#include <random>
#include <map>
#include <thread>
#include <sys/sysinfo.h>

#define C_NRML "\033[0m" 
#define C_BLCK "\033[30m"
#define C_RED  "\033[31m"
#define C_GREN "\033[32m"
#define C_YLLW "\033[33m"
#define C_BLUE "\033[34m"
#define C_PRPL "\033[35m"
#define C_AQUA "\033[36m"

typedef struct TestParam{
  double interval;
  int gpu_cycle;
  int cpu_cycle;
}TestParam;

class Workload {
 public:
  Workload();
  // Workload(int duration, int cpu, int gpu, bool random);
  Workload(int single_test_duration,
           int init_wait_time,
           std::string offset_file_name,
           std::string param_file_name);

  ~Workload();

  void GPU_Worker();
  void CPU_Worker();

  void CPUWorkload();
  void GPUWorkload();

  int ReadOffsets(std::string& offset_file_name);
  int ReadParams(std::string& param_file_name);

  float CalculateOffsetandDutyCycle(int resource);

 private:

  float cpugpu_transition;
  int gpu_kernel_size;
  int cpu_cores;

  struct timespec start_time;
  bool ignition = false;
  bool cpu_ignition = false;
  bool gpu_ignition = false;
  bool gpu_kernel_done = false;
  bool terminate = false;

  bool cpu_workload_terminate = false;
  bool gpu_workload_terminate = false;

  std::thread cpu_workload;
  std::thread gpu_workload;

  std::atomic<int> cpu_inner_test_sequence_count;
  std::atomic<int> gpu_inner_test_sequence_count;
  std::atomic<int> global_test_sequence_count;
  std::atomic<int> global_inner_test_sequence;

  std::atomic<float> cpu_workload_duty_cycle;
  std::atomic<float> gpu_workload_duty_cycle;
  std::atomic<float> workload_interval;

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
  int single_test_duration;
  int init_wait_time;

  std::vector<TestParam> test_params;
  
  std::vector<std::pair<int, int>> offsets;
  std::string offset_file_name;
  std::string param_file_name;

  // GPU workload pool
  std::vector<std::thread> gpu_workload_pool;

  // CPU workload pool
  std::vector<std::thread> cpu_workload_pool;

};  // class Workload