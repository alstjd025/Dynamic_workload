#include "dynamic_workload.h"
#define GPU_UTIL_FILE "/mnt/ramdisk/gpu_util"

#define GPU_KERNEL_SIZE 15

const char* computeShaderSource = R"(
#version 310 es
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in; // Work group size

layout(std430, binding = 0) readonly buffer MatrixA {
    float A[];
};

layout(std430, binding = 1) readonly buffer MatrixB {
    float B[];
};

layout(std430, binding = 2) writeonly buffer MatrixC {
    float C[];
};

uniform ivec3 sizeA; // (x1, y1, z1)
uniform ivec3 sizeB; // (x2, y2, z2)

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID); // Global ID for each thread

    int x1 = sizeA.x;
    int y1 = sizeA.y;
    int z1 = sizeA.z;

    int x2 = sizeB.x;
    int y2 = sizeB.y;
    int z2 = sizeB.z;

    // Ensure valid multiplication indices
    if (gid.x >= x1 || gid.y >= y2 || gid.z >= z2) {
        return;
    }

    float sum = 0.0;
    for (int i = 0; i < y1; ++i) { // y1 == x2 for matrix multiplication compatibility
        int indexA = gid.x * (y1 * z1) + i * z1 + gid.z;
        int indexB = i * (y2 * z2) + gid.y * z2 + gid.z;
        sum += A[indexA] * B[indexB];
    }

    int indexC = gid.x * (y2 * z2) + gid.y * z2 + gid.z;
    C[indexC] = sum;
}
)";

bool m_break = false;

void INThandler(int sig) {
  signal(sig, SIG_IGN);
  m_break = true;
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
  cpugpu_transition = 0;
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
  clock_gettime(CLOCK_MONOTONIC, &begin);
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
  std::cout << "CPU offset " << offset << " duty " << cpu_duty_cycle << " interval " << interval << "\n";
  // wait for offset time.
  std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(offset)));
  printf("%s CPU duty cycle start, works for %dms %s \n", C_GREN, static_cast<int>(cpu_duty_cycle), C_NRML);
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
  // printf("CPU elapsed %.6fs\n", elapsed_t);
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
  std::cout << "GPU offset " << offset << " duty " << gpu_duty_cycle << " interval " << interval << "\n";
  // wait for offset time.
  std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(offset)));
  printf("%s GPU duty cycle start, works for %dms %s \n", C_GREN, static_cast<int>(gpu_duty_cycle), C_NRML);
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
  EGLDisplay display;
  EGLContext context;
  EGLSurface surface;
  int count=1, idx;
  double response_t = 0;
  double tot_response_t = 0;
  struct timespec begin, end;

  // Initialize EGL
  display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  // if (display == EGL_NO_DISPLAY) {
  //   printf("eglGetDisplay returned EGL_NO_DISPLAY.\n");
  //   return;
  // }
  EGLBoolean returnValue = eglInitialize(display, NULL, NULL);
  // if (returnValue != EGL_TRUE) {
  //   printf("eglInitialize failed\n");
  //   return;
  // }
  // Configure EGL attributes
  EGLConfig config;
  EGLint numConfigs;
  EGLint configAttribs[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_NONE};
  eglChooseConfig(display, configAttribs, &config, 1, &numConfigs);

  // Create an EGL context
  EGLint contextAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};

  context = eglCreateContext(display, EGL_NO_CONTEXT, EGL_CAST(EGLConfig, 0),
                             contextAttribs);
  // if (context == EGL_NO_CONTEXT) {
  //   printf("eglCreateContext failed\n");
  //   return;
  // }
  // Create a surface
  surface = eglCreatePbufferSurface(display, config, NULL);

  // Make the context current
  eglMakeCurrent(display, surface, surface, context);
  // if (returnValue != EGL_TRUE) {
  //   printf("eglMakeCurrent failed returned %d\n", returnValue);
  //   return;
  // }
  // Compile compute shader
  GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
  glShaderSource(computeShader, 1, &computeShaderSource, NULL);
  glCompileShader(computeShader);

  // Create program and attach shader
  GLuint program = glCreateProgram();
  glAttachShader(program, computeShader);
  glLinkProgram(program);
  GLint linkStatus = GL_FALSE;
  glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
  // if (!linkStatus) {
  //   printf("glGetProgramiv failed returned \n");
  //   return;
  // }

  // Initialize data
  // computation
  int x1 = 1024, y1 = 128, z1 = 256; // Matrix A size (4x4x4)
  int x2 = 32, y2 = 32, z2 = gpu_kernel_size; // Matrix B size (4x4x4)
  
  // nano                nx
  // z2 512
  // z2 412              202ms 
  // z2 256              128ms
  // z2 128 459ms        
  // z2 55 201ms
  // z2 29 105
  // z2 27 99ms
  // z2 15 50ms
  // z2 3 11mss
  // z2 6 20 ms
  // Initialize matrices A and B with some data
  std::vector<float> A(x1 * y1 * z1, 1.0f); // Fill with 1.0f for simplicity
  std::vector<float> B(x2 * y2 * z2, 2.0f); // Fill with 2.0f for simplicity
  std::vector<float> C(x1 * y2 * z2, 0.0f); // Result matrix initialized to 0.0f

  // Create buffer objects
  GLuint bufferA, bufferB, bufferC;
  glGenBuffers(1, &bufferA);
  glGenBuffers(1, &bufferB);
  glGenBuffers(1, &bufferC);


  glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferA);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferB);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferC);


  glBufferData(GL_SHADER_STORAGE_BUFFER, A.size() * sizeof(float),
               A.data(), GL_STATIC_DRAW);
  glBufferData(GL_SHADER_STORAGE_BUFFER, B.size() * sizeof(float),
               B.data(), GL_STATIC_DRAW);
  glBufferData(GL_SHADER_STORAGE_BUFFER, C.size() * sizeof(float),
               C.data(), GL_STATIC_DRAW);
              
  

  // Bind buffer objects to binding points
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferA);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferB);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, bufferC);
  
  glUniform3i(glGetUniformLocation(program, "sizeA"), x1, y1, z1);
  glUniform3i(glGetUniformLocation(program, "sizeB"), x2, y2, z2);

  glUseProgram(program);
  std::cout << "Created new GPU worker \n";
  while(!gpu_worker_termination){
    {
      std::unique_lock<std::mutex> lock_(gpu_mtx);
      gpu_cv.wait(lock_, [this]() { return gpu_ignition; });
    }

    signal(SIGINT, INThandler);
    std::ofstream outfile;

    // multi-level test
    // Todo :
    count = 0;
    float gpu_elapsed_t = 0;
    struct timespec seq_begin;
    clock_gettime(CLOCK_MONOTONIC, &seq_begin);
    // std::cout << "gpu go" << "\n";
    while (!gpu_stop) {
      if (m_break) break;
      // int PERIOD = 5;
      // glDispatchCompute(16, 16, 1);

      // std::this_thread::sleep_for(std::chrono::milliseconds(PERIOD));
      
      clock_gettime(CLOCK_MONOTONIC, &begin);
      glDispatchCompute((GLuint)x1, (GLuint)y2, (GLuint)z2);
      glFlush();  // Ensures that the dispatch command is processed, delete
      // // Create a fence sync object and wait for the GPU to finish
      GLsync syncObj = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0); //delete
      glWaitSync(syncObj, 0, GL_TIMEOUT_IGNORED); // delete 
      clock_gettime(CLOCK_MONOTONIC, &end);

      response_t = (end.tv_sec - begin.tv_sec) +
                   ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
      // tot_response_t += response_t;
      count++;

      //glDeleteSync(syncObj);  // Clean up the sync object
      glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
      glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
      glFinish();  // all commmand push to GPU HW queue (gpu has two queue, gpu
                  // drvier queue + gpu hw queue )
      // printf("gpu kernel latency : %.11f\n", response_t);
      clock_gettime(CLOCK_MONOTONIC, &end);
      gpu_elapsed_t = (end.tv_sec - seq_begin.tv_sec) +
                  ((end.tv_nsec - seq_begin.tv_nsec) / 1000000000.0);
      if (gpu_elapsed_t > gpu_workload_duty_cycle) {
        gpu_stop = true;
      }
    }
    // wake main thread
    {
      std::unique_lock<std::mutex> lock_data(gpu_mtx);
      gpu_kernel_done = true;
      gpu_ignition = false;
      gpu_end_cv.notify_one();
    }
  // printf("%d's average : %.11f\n", count, (tot_response_t / double(count)));
  }

  // Read back result
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferC);
  float* output = (float*)(glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
                                            sizeof(float) * C.size(),
                                            GL_MAP_READ_BIT));

  // Clean up
  glDeleteShader(computeShader);
  glDeleteProgram(program);
  glDeleteBuffers(1, &bufferA);
  glDeleteBuffers(1, &bufferB);
  glDeleteBuffers(1, &bufferC);

  // Tear down EGL
  eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
  eglDestroySurface(display, surface);
  eglDestroyContext(display, context);
  eglTerminate(display);

  std::cout << "Terminates GPU worker "
            << "\n";
  return;
}

Workload::~Workload(){};
