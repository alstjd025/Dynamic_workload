#include "dynamic_workload.h"
#define GPU_UTIL_FILE "/home/odroid/Dynamic_workload/gpu_util"

#define GPU_KERNEL_SIZE 256

const char* computeShaderSource = R"(
  __kernel void matmul(
      __global const float* A,
      __global const float* B,
      __global float* C,
      const unsigned int M,
      const unsigned int N,
      const unsigned int P) {

      int row = get_global_id(0);
      int col = get_global_id(1);

      if (row < M && col < P) {
          float sum = 0.0f;
          for (int k = 0; k < N; ++k) {
              sum += A[row * N + k] * B[k * P + col];
          }
          C[row * P + col] = sum;
      }
  })";

bool m_break = false;

void INThandler(int sig) {
  signal(sig, SIG_IGN);
  m_break = true;
}

void WriteUtilization(int utilization, int workload){
  std::ofstream gpu_util_file(GPU_UTIL_FILE);
    if (gpu_util_file.is_open()) {
        gpu_util_file << utilization << " " << workload;
        gpu_util_file.close();
    } else {
        std::cerr << "Failed to open gpu_util file for writing.\n";
    }
}

Workload::Workload(){};

Workload::Workload(int single_test_duration_,
                   int init_wait_time_, 
                   std::string offset_file_name_,
                   std::string param_file_name_) {
  struct timespec init, begin, end, begin_i, end_i;
  /* Total execution occurs in duration x size (sec)*/
  offset_file_name = offset_file_name_;
  if(ReadOffsets(offset_file_name) != 1){
    std::cout << "Offset file read error" << "\n";
    return;
  }
  if(ReadParams(param_file_name_) != 1){
    std::cout << "Param file read error" << "\n";
    return;
  }
  int size = 1;

  // no need

  single_test_duration = single_test_duration_;
  init_wait_time = init_wait_time_;
  gpu_kernel_size = GPU_KERNEL_SIZE;
  cpu_cores = get_nprocs();

  std::cout << "Dynamic dummy workload" << "\n";
  std::cout << "Single test duration: " << single_test_duration << "\n";
  std::cout << "Inital wait time: " << init_wait_time << "s \n";
  std::cout << "GPU kernel size: " << gpu_kernel_size << "\n";
  std::cout << "Number of CPU coers: " << cpu_cores << "\n" ;
  std::cout << "Number of total test sequences: " << test_params.size() << "\n";

  std::cout << C_GREN << "========Workload Init=========\n" << C_NRML;
  ///////////////////////////////////////////////////////////////////////
  ////// workload start
  cpu_inner_test_sequence_count = 0;
  gpu_inner_test_sequence_count = 0; 
  global_inner_test_sequence = 0;
  global_test_sequence_count = 0;
  cpugpu_transition = 0;
  double elapsed_t = 0;
  double total_elapsed_t = 0;
  double interval_elapsed_t = 0;
  double single_interval = 0;
  int maximum_test = 0;
  cpu_workload_pool.reserve(cpu_cores);
  stop = false;
  cpu_worker_termination = false;
  gpu_worker_termination = false;
  for (int i = 0; i < cpu_cores; ++i) {
    std::cout << "Creates " << i << " cpu worker"
              << "\n";
    cpu_workload_pool.emplace_back([this]() { this->CPU_Worker(); });
  }
  //Minsung
  gpu_workload_pool.reserve(1);
  for (int i = 0; i < 1; ++i) {
    std::cout << "Creates " << i << " gpu worker"
              << "\n";
    cpu_workload_pool.emplace_back([this]() { this->GPU_Worker(); });
  }
  std::cout << "Creates kernel size " << gpu_kernel_size << " GPU worker"
            << "\n";
  gpu_workload_pool.emplace_back([this]() { this->GPU_Worker(); });
  
  double elapsed_t_millisec = 0;
  // Wait for inital waiting time.
  std::this_thread::sleep_for(std::chrono::seconds(init_wait_time));
  std::cout << C_GREN << "========Workload start=========\n" << C_NRML;

  while(global_test_sequence_count < test_params.size()){
    maximum_test = single_test_duration / test_params[global_test_sequence_count].interval;
    if(maximum_test > offsets.size()){
      std::cout << C_RED << "Dynamic workload: maximum test sequence exceeds offset params"
                         <<  " begin ====\n" <<C_NRML;
    }
    while(global_inner_test_sequence < maximum_test){
      clock_gettime(CLOCK_MONOTONIC, &init);
      std::cout << C_GREN << "==== Workload sequence: " << global_inner_test_sequence + 1 
                <<  "/"<< maximum_test << " begin ====\n" <<C_NRML;
      // CPU and GPU worklaod should work in single interval.
      // start CPU worker
      cpu_workload = std::thread(&Workload::CPUWorkload, this);  
      // start GPU worker
      gpu_workload = std::thread(&Workload::GPUWorkload, this);  
      
      cpu_workload.join();
      gpu_workload.join();
      clock_gettime(CLOCK_MONOTONIC, &end);
      elapsed_t_millisec = (end.tv_sec * 1000.0 - init.tv_sec * 1000.0) +
            ((end.tv_nsec - init.tv_nsec) / 1000000.0);
      std::cout << C_GREN << "==== Workload sequence: " << global_inner_test_sequence + 1
                <<  "/" << maximum_test << " end " << 
                static_cast<int>(elapsed_t_millisec) << "ms ===\n" <<C_NRML;
      cpu_inner_test_sequence_count += 1;
      gpu_inner_test_sequence_count += 1;
      global_inner_test_sequence += 1;
    }
    cpu_inner_test_sequence_count = 0;
    gpu_inner_test_sequence_count = 0; 
    global_inner_test_sequence = 0;
    global_test_sequence_count += 1;
    // Calaculate timing???
  }
  
  ////// workload end
  ///////////////////////////////////////////////////////////////////////

  // CPU worker kill
  cpu_worker_termination = true;
  cpu_stop = true;
  {  // wakes  workers
    std::unique_lock<std::mutex> lock(cpu_mtx);
    cpu_ignition = true;
    cpu_cv.notify_all();
    std::cout << "Notified all CPU workers to kill"
              << "\n";
  }
  gpu_worker_termination = true;
  gpu_stop = true;
  {  // wakes  workers
    std::unique_lock<std::mutex> lock(gpu_mtx);
    gpu_ignition = true;
    gpu_cv.notify_all();
    std::cout << "Notified GPU workers to kill"
              << "\n";
  }
  stop = true;
  ignition = false;
  for (auto& workers : gpu_workload_pool) workers.join();
  for (auto& workers : cpu_workload_pool) workers.join();
  cpu_workload_pool.clear();
  gpu_workload_pool.clear();
  std::cout << "====== Workload done ======\n";
};

void Workload::CPUWorkload(){
  struct timespec begin, end;
 // clock_gettime(CLOCK_MONOTONIC, &begin);
  float elapsed_t_millisec;

  // calculate offset and duty cycle
  float offset = CalculateOffsetandDutyCycle(1);  // 1 means CPU
  // cpu_workload_duty_cycle is calculated from CalculateOffsetandDutyCycle().
  float cpu_duty_cycle = cpu_workload_duty_cycle;
  float interval = workload_interval;
  // workload for single interval

  // we calculate every timing in sec. so change sec to millisec here.
  offset *= 1000.0; // change offset to millisec (ex, 0.7 sec -> 700ms)
  cpu_duty_cycle *= 1000.0; // change duty cycle to millisec (ex, 0.5 sec -> 500ms)
  interval *= 1000.0; // change interval cycle to millisec (ex, 1 sec -> 1000ms)
  //std::cout << "CPU offset " << offset << " duty " << cpu_duty_cycle << " interval " << interval << "\n";
  // wait for offset time.
  std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(offset)));
  printf("%s CPU duty cycle start, works for %dms %s \n", C_GREN, static_cast<int>(cpu_duty_cycle), C_NRML);
  clock_gettime(CLOCK_MONOTONIC, &begin);
	cpu_stop = false;
  {  // wakes  workers
    std::unique_lock<std::mutex> lock(cpu_mtx);
    cpu_ignition = true;
    cpu_cv.notify_all();
  }
  elapsed_t_millisec = 0;
  // do work for duty cycle
  std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(cpu_duty_cycle)));
  cpu_stop = true;
  cpu_ignition = false;
  printf("%s CPU duty cycle end %s \n", C_GREN, C_NRML);
  clock_gettime(CLOCK_MONOTONIC, &end);
  elapsed_t_millisec = (end.tv_sec * 1000.0 - begin.tv_sec * 1000.0) +
        ((end.tv_nsec - begin.tv_nsec) / 1000000.0);
   printf("CPU elapsed %.6fs\n", elapsed_t_millisec);
  // stop work 
  float eta = interval - elapsed_t_millisec;
  if(eta > 0){
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(eta)));
  }
  // std::cout << "CPU workload done" << "\n";
}

void Workload::GPUWorkload(){
  struct timespec begin, end;
  clock_gettime(CLOCK_MONOTONIC, &begin);
  double elapsed_t_millisec;

  // calculate offset and duty cycle.
  float offset = CalculateOffsetandDutyCycle(2);  // 2 means GPU
  // cpu_workload_duty_cycle is calculated from CalculateOffsetandDutyCycle().
  float gpu_duty_cycle = gpu_workload_duty_cycle;
  float interval = workload_interval;

  // we calculate every timing in sec. so change sec to millisec here.
  offset *= 1000.0; // change offset to millisec (ex, 0.7 sec -> 700ms)
  gpu_duty_cycle *= 1000.0; // change duty cycle to millisec (ex, 0.5 sec -> 500ms)
  interval *= 1000.0; // change interval cycle to millisec (ex, 1 sec -> 1000ms)
  //std::cout << "GPU offset " << offset << " duty " << gpu_duty_cycle << " interval " << interval << "\n";
  // wait for offset time.
  std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(offset)));
  printf("%s GPU duty cycle start, works for %dms %s \n", C_GREN, static_cast<int>(gpu_duty_cycle), C_NRML);
  WriteUtilization(100, 1);
  gpu_stop = false;
  {  // wakes  workers
    std::unique_lock<std::mutex> lock(gpu_mtx);
    gpu_ignition = true;
    gpu_kernel_done = false;
    gpu_cv.notify_all();
  }
  elapsed_t_millisec = 0;
  { // GPU kernel return wait
    std::unique_lock<std::mutex> lock_data(gpu_mtx);
    gpu_end_cv.wait(lock_data, [&] { return gpu_kernel_done; });
  }
  WriteUtilization(0, 0);
  printf("%s GPU duty cycle end %s \n", C_GREN, C_NRML);
  gpu_stop = true;
  clock_gettime(CLOCK_MONOTONIC, &end);
  elapsed_t_millisec = (end.tv_sec * 1000.0 - begin.tv_sec * 1000.0) +
        ((end.tv_nsec - begin.tv_nsec) / 1000000.0);

  float eta = interval - elapsed_t_millisec;
  if(eta > 0){
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(eta)));
  }
}

float Workload::CalculateOffsetandDutyCycle(int resource){
  float interval = 0;
  float duty_cycle = 0;
  float offset = 0;
  int offset_percentage = 0;
  int inner_test_sequence = 0;
  int test_sequence = global_test_sequence_count;
  if(test_params.empty() || offsets.empty()){
    std::cout  << C_RED << "DynamicWorkload: offset calculation failed,"
               << "empty test params or offset params." 
               << C_NRML << "\n";
    return -1;
  }
  if(test_params.size() == 1 && test_sequence > 0 ){
    test_sequence = 0;
  }
  interval = test_params[test_sequence].interval;
  workload_interval = interval;
  if(resource == 1){ // 1 means CPU
    inner_test_sequence = cpu_inner_test_sequence_count;
    offset_percentage = offsets[inner_test_sequence].first;
    duty_cycle = test_params[test_sequence].cpu_cycle / 100.0;
    cpu_workload_duty_cycle = interval * duty_cycle;
    offset = (((interval - cpu_workload_duty_cycle) / 100.0) * offset_percentage);
  }else if(resource == 2){ // 2 means GPU
    inner_test_sequence = gpu_inner_test_sequence_count;
    offset_percentage = offsets[inner_test_sequence].second;
    duty_cycle = test_params[test_sequence].gpu_cycle / 100.0;
    gpu_workload_duty_cycle = interval * duty_cycle;
    offset = (((interval - gpu_workload_duty_cycle) / 100.0) * offset_percentage);
  }else{
    std::cout  << C_RED << "DynamicWorkload:"
               << " offset calculation failed, wrong resource." 
               << C_NRML << "\n";
  }
  // calculate offset.
  return offset;
}

int Workload::ReadOffsets(std::string& offset_file_name){
  std::ifstream inFile(offset_file_name);
  if (!inFile) {
      std::cerr << "Cannot open offset file." << std::endl;
      return -1;
  }

  std::vector<std::pair<int, int>> data;
  int num1, num2;
  while (inFile >> num1 >> num2) {
      offsets.emplace_back(num1, num2); // pair를 벡터에 추가
  }
  inFile.close();
  return 1;
}

int Workload::ReadParams(std::string& param_file_name){
  std::ifstream inFile(param_file_name);
  if (!inFile) {
      std::cerr << "파일을 열 수 없습니다." << std::endl;
      return 0;
  }

  double interval;
  int gpu_cycle, cpu_cycle;

  // 파일에서 데이터를 읽어 구조체에 저장
  while (inFile >> interval >> gpu_cycle >> cpu_cycle) {
    TestParam param{interval, gpu_cycle, cpu_cycle}; // 구조체 초기화
    test_params.push_back(param); // 벡터에 구조체 추가
  }

  inFile.close();

  // 데이터 확인 출력
  // for (const auto& param : test_params) {
  //   std::cout << "Interval: " << param.interval
  //             << ", GPU Cycle: " << param.gpu_cycle
  //             << ", CPU Cycle: " << param.cpu_cycle << std::endl;
  // }

  return 1;
}

void Workload::CPU_Worker() {
  // not implemented
  while(!cpu_worker_termination){
    // std::cout << "cpu worker start" << "\n";
    {
      std::unique_lock<std::mutex> lock_(cpu_mtx);
      cpu_cv.wait(lock_, [this]() { return cpu_ignition; });
    }
    double a = 1;
    double b = 0.0003;
    while (!cpu_stop) {
      a *= b;
    }
  }
  std::cout << "Terminates CPU worker " << "\n";
};


void Workload::GPU_Worker() {
  struct timespec init_begin, init_end;
  clock_gettime(CLOCK_MONOTONIC, &init_begin);

  try {
      // OpenCL 플랫폼, 디바이스, 큐 설정
      std::vector<cl::Platform> platforms;
      cl::Platform::get(&platforms);
      if (platforms.empty()) throw std::runtime_error("No OpenCL platforms found.");

      auto platform = platforms.front();
      std::vector<cl::Device> devices;
      platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
      if (devices.empty()) throw std::runtime_error("No GPU devices found on the platform.");

      auto device = devices.front();
      std::cout << "Successfully retrieved platform and device.\n";

      cl::Context context(device);
      cl::CommandQueue queue(context, device);
      cl::Program program(context, computeShaderSource);

      program.build("-cl-std=CL1.2");
      std::cout << "OpenCL kernel compiled successfully.\n";

      // 버퍼 및 행렬 크기 초기화
      const int x1 = 1024, y1 = 128, z1 = 256;
      const int x2 = 32, y2 = 32, z2 = gpu_kernel_size;
      const int matrixElements = x1 * y2 * z2;
      std::vector<float> matrixA(x1 * y1 * z1);
      std::vector<float> matrixB(matrixElements);
      std::vector<float> resultMatrix(matrixElements);

      for (int i = 0; i < matrixElements; ++i) {
          matrixA[i] = static_cast<float>(i);
          matrixB[i] = static_cast<float>(i + matrixElements);
          resultMatrix[i] = static_cast<float>(0);
      }

      // 버퍼 할당
      cl::Buffer bufferA(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * x1 * y1 * z1, matrixA.data());
      cl::Buffer bufferB(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * x2 * y2 * z2, matrixB.data());
      cl::Buffer bufferResult(context, CL_MEM_READ_WRITE, sizeof(float) * matrixElements);
      
      cl::Kernel kernel(program, "matmul");  // 커널 이름을 matmul로 변경
      kernel.setArg(0, bufferA);
      kernel.setArg(1, bufferB);
      kernel.setArg(2, bufferResult);
      kernel.setArg(3, x1);  // M (행 수)
      kernel.setArg(4, y1);  // N (내적 크기)
      kernel.setArg(5, z2);  // P (열 수)

      // 버퍼 및 커널 인수 설정 확인
      std::cout << "Buffers and kernel arguments initialized successfully.\n";

      queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(float) * x1 * y1 * z1, matrixA.data());
      queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, sizeof(float) * x2 * y2 * z2, matrixB.data());

      signal(SIGINT, INThandler);
      clock_gettime(CLOCK_MONOTONIC, &init_end);
      double init_time = (init_end.tv_sec - init_begin.tv_sec) + ((init_end.tv_nsec - init_begin.tv_nsec) / 1e9);
      printf("init time : %.11f\n", init_time);

      struct timespec begin, end;
      std::cout << "Ready to perform matrix multiplication.\n";

      while (!gpu_worker_termination) {
          int count = 0;
          double tot_response_t = 0.0, gpu_elapsed_t = 0.0;
          struct timespec seq_begin;

          void* mapped_ptr_A = queue.enqueueMapBuffer(bufferA, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * x1 * y1 * z1);
          if (mapped_ptr_A == nullptr) throw std::runtime_error("Failed to map buffer A.");

          void* mapped_ptr_B = queue.enqueueMapBuffer(bufferB, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * x2 * y2 * z2);
          if (mapped_ptr_B == nullptr) throw std::runtime_error("Failed to map buffer B.");

          {
              std::unique_lock<std::mutex> lock_(gpu_mtx);
              gpu_cv.wait(lock_, [this]() { return gpu_ignition; });
          }

          clock_gettime(CLOCK_MONOTONIC, &seq_begin);

          while (!gpu_stop) {
              if (m_break) break;

              clock_gettime(CLOCK_MONOTONIC, &begin);
              // 커널 실행: 행렬 곱셈을 위한 NDRange 설정
              cl_int err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(x1, z2), cl::NDRange(1, 1), NULL, NULL);
              //std::cout << err << std::endl;
              if (err != CL_SUCCESS) {
                  std::cerr << "Failed to enqueue NDRange kernel, error code: " << err << "\n";
                  throw std::runtime_error("Kernel execution failed.");
              }
              queue.finish();

              clock_gettime(CLOCK_MONOTONIC, &end);
              double response_t = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1e9);
              gpu_elapsed_t += response_t;
              //printf("response_t time : %.11f\n", response_t);
              if (gpu_elapsed_t > gpu_workload_duty_cycle) {
                  printf("gpu elapsed time : %.11f\n", gpu_elapsed_t);
                  gpu_stop = true;
              }
              count++;
          }

          {
              std::unique_lock<std::mutex> lock_data(gpu_mtx);
              gpu_kernel_done = true;
              gpu_ignition = false;
              gpu_end_cv.notify_one();
          }

          queue.enqueueReadBuffer(bufferResult, CL_TRUE, 0, sizeof(float) * matrixElements, resultMatrix.data());
          //std::cout << "Result matrix successfully read back from GPU.\n";

          queue.enqueueUnmapMemObject(bufferA, mapped_ptr_A);
          queue.enqueueUnmapMemObject(bufferB, mapped_ptr_B);
          //std::cout << "Buffers unmapped successfully.\n";
      }

      std::cout << "GPU worker terminated.\n";
  }
  catch (const std::exception& e) {
      std::cerr << "Exception: " << e.what() << std::endl;
  }

  std::cout << "Terminates GPU worker "
            << "\n";
  return;
}

Workload::~Workload(){};
