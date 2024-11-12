#include <iostream>
#include <fstream>
#include <vector>
#include <utility> // std::pair 사용을 위해 필요

int main() {
    std::ifstream inFile("offset");
    if (!inFile) {
        std::cerr << "파일을 열 수 없습니다." << std::endl;
        return 1;
    }

    std::vector<std::pair<int, int>> data;
    int num1, num2;
    while (inFile >> num1 >> num2) {
        data.emplace_back(num1, num2); // pair를 벡터에 추가
    }

    inFile.close();

    // 읽어온 데이터 확인 출력
    for (const auto& pair : data) {
        std::cout << pair.first << " " << pair.second << std::endl;
    }

    return 0;
}
