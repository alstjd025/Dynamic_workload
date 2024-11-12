#include "dynamic_workload.h"

int main(int argv, char* argc[]){
  if(argv < 4){
    std::cout << "Not enough args, usage : duration(sec), initial wait time(sec)" <<
                 " offset file path, parameter file path" << "\n";
    exit(-1);
  }
  int duration = atoi(argc[1]);
  int wait_time = atoi(argc[2]);

  // Workload workload(duration, cpu, gpu, false);
  Workload workload(duration, wait_time, std::string(argc[3]), std::string(argc[4]));

  return 0;
}