CC=clang++
CFLAGS=-std=c++20 -Wall -Wextra -O2
LDFLAGS=-g -lvulkan -ldl -lpthread -lm -L../vulkan_tuto/glfw_lib/build/src/ -lglfw3 #Las ultimas dos son lo que son XD
INLCUDE_DIRS=vendor/
SRC_FILES=src/main.cpp

.PHONY: all clean

all: vulkan

vulkan: $(SRC_FILES)
	$(CC) $(CFLAGS) -I$(INLCUDE_DIRS) -o $@ $^ $(LDFLAGS)
	
clean:
	rm -f vulkan
