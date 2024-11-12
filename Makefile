CC := g++
CFLAGS := -std=c++11 -I/usr/include
LDFLAGS := -lpthread -lEGL -lGLESv2

SRCS := dummy_workload.cpp main.cpp 
GL_SRCS := gl_test.cpp

OBJS := $(SRCS:.cpp=.o)
GL_OBJS := $(GL_SRCS:.cpp=.o)

EXEC := dummy_workload
GL_EXEC := gl_test

.PHONY: all clean

dummy: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

gl_test: $(GL_EXEC)

$(GL_EXEC): $(GL_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXEC) $(GL_OBJS) $(GL_EXEC)
