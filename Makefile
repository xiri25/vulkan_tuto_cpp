CC=clang++
CFLAGS=-std=c++20 -Wall -Wextra -O0
LDFLAGS=-g -lvulkan -ldl -lpthread -lm -Lvendor/glfw/build/src/ -lglfw3 #Las ultimas dos son lo que son XD
INLCUDE_DIRS=vendor/glfw/include/ \
			 vendor/glm/ \
			 vendor/tinyobjloader/ \
			 vendor/stb/
SRC_FILES=src/main.cpp

.PHONY: all clean

all: vulkan

vulkan: $(SRC_FILES)
	$(CC) $(CFLAGS) $(addprefix -I,$(INLCUDE_DIRS)) -o $@ $^ $(LDFLAGS)
	
clean:
	rm -f vulkan
