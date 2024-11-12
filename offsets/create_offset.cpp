#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

int main() {
    std::ofstream outFile("offset");
    if (!outFile) {
        std::cerr << "파일을 열 수 없습니다." << std::endl;
        return 1;
    }

    std::srand(static_cast<unsigned int>(std::time(0))); // 시드 초기화
    for (int i = 0; i < 360; ++i) {
        int num1 = std::rand() % 101; // 0~100 사이의 정수 생성
        int num2 = std::rand() % 101;
        outFile << num1 << " " << num2 << "\n";
    }

    outFile.close();
    std::cout << "파일이 성공적으로 생성되었습니다." << std::endl;
    return 0;
}
