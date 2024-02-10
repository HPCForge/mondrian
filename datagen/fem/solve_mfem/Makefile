CC = g++
OPT_FLAGS ?=
FLAGS = -std=c++17 $(OPT_FLAGS)


SRC_DIR = solve
BUILD_DIR = build

SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OUT_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.out,$(SRC_FILES))

MFEM_PATH ?= /pub/afeeney/opensource/mfem/mfem/
MFEM = $(MFEM_PATH)/libmfem.a -I$(MFEM_PATH)

all: build_dir $(OUT_FILES) 

$(BUILD_DIR)/%.out: $(SRC_DIR)/%.cpp
	$(CC) $(FLAGS) -o $@ $< $(MFEM)

build_dir:
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
