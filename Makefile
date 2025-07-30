.PHONY: all build run clean see nsys ncu

# executable
BUILD_DIR = build
EXECUTABLE_NAME = image_conv

# input/output images
INPUT=input/pebble.pgm
OUTPUT=output/pebble_blurred.pgm

# profiling
REPORT_NCU=profile/report_ncu

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake ..
	@cd $(BUILD_DIR) && cmake --build .

run:
	$(BUILD_DIR)/$(EXECUTABLE_NAME) ${INPUT} ${OUTPUT}

see:
	gimp ${OUTPUT}

# profiling
ncu:
	ncu -o ${REPORT_NCU} \
	$(BUILD_DIR)/$(EXECUTABLE_NAME) ${INPUT} ${OUTPUT}

# clean up
clean:
	@rm -rf $(BUILD_DIR) output profile
