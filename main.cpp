#include "dynamic_workload.h"

int main(int argv, char* argc[]){
  if(argv < 4){
    std::cout << "Not enough args, usage : duration(0~), transition time(), kernel size() " <<
                 "num_cpu(0~)" << "\n";
    exit(-1);
  }
  int cpu, gpu, duration;
  float transition_time;
  int kernel_size;

  duration = atoi(argc[1]);
  transition_time = stof(std::string(argc[2]));
  kernel_size = atoi(argc[3]);
  cpu = atoi(argc[4]);
  // gpu = atoi(argc[3]);
  
  
  

  // Workload workload(duration, cpu, gpu, false);
  Workload workload(duration, transition_time, kernel_size, cpu, false);

  return 0;
}