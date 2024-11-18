#include "dynamic_workload.h"

int main(int argv, char* argc[]){
  if(argv < 2){
    std::cout << "Not enough args, usage : parameter file path" << "\n";
    exit(-1);
  }
  std::cout << argc[1] << "\n";
  std::string filename = argc[1];
  // Workload workload(duration, cpu, gpu, false);
  Workload workload(filename);
  std::cout << "hello" << "\n";
  return 0;
}