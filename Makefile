CC := g++
CFLAGS := -std=c++11 -I/usr/include 
LDFLAGS := -lpthread -lEGL -lGLESv2 

SRCS := dynamic_workload.cpp main.cpp 
GL_SRCS := gl_test.cpp

OBJS := $(SRCS:.cpp=.o)
GL_OBJS := $(GL_SRCS:.cpp=.o)

EXEC := dynamic_workload

.PHONY: all clean

dummy: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJS) $(EXEC) $(GL_OBJS)
