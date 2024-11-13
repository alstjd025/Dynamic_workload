#include <iostream>
#include <fstream>
#include <vector>

int main() {
    // 변수 벡터 초기화
    std::vector<int> interval = {10, 5, 1}; //refers to sec
    std::vector<int> gpu_cycle = {75, 50, 25}; // % of cycle
    std::vector<int> cpu_cycle = {75, 50, 25}; // % of cycle

    // 파일 출력 스트림 생성
    std::ofstream outFile("params");
    if (!outFile) {
        std::cerr << "cannot open param file." << std::endl;
        return 1;
    }

    // interval, gpu_cycle, cpu_cycle의 모든 조합 생성
    for (int i : interval) {
        for (int j : gpu_cycle) {
            for (int k : cpu_cycle) {
                outFile << i << " " << j << " " << k << "\n";
            }
        }
    }

    outFile.close();
    std::cout << "params saved." << std::endl;

    return 0;
}
