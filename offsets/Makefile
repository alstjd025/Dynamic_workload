# 컴파일러
CXX = g++

# 컴파일 옵션
CXXFLAGS = -std=c++11 -Wall

# 실행 파일 이름
TARGET = generate_offsets

# 빌드 규칙
all: $(TARGET)

$(TARGET): create_offset.o
	$(CXX) $(CXXFLAGS) -o $(TARGET) create_offset.o

create_offset.o: create_offset.cpp
	$(CXX) $(CXXFLAGS) -c create_offset.cpp

# 클린 규칙
clean:
	rm -f $(TARGET) *.o offset
