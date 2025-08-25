.PHONY: all build run see profile run_npp see_npp profile_npp clean

# executable
BUILD_DIR = build
EXECUTABLE_NAME_NPP = image_conv_npp
EXECUTABLE_NAME_KERNEL = image_conv_kernel


# input/output
INPUT=input/pebble.pgm
OUTPUT_NPP=output/pebble_blurred_npp.pgm
OUTPUT_KERNEL=output/pebble_blurred_kernel

REPORT_NCU_KERNEL=profile/report_ncu_kernel
REPORT_NCU_NPP=profile/report_ncu_npp

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake ..
	@cd $(BUILD_DIR) && cmake --build .

# benchmark
run_npp:
	$(BUILD_DIR)/$(EXECUTABLE_NAME_NPP) ${INPUT} ${OUTPUT_NPP}

see_npp:
	gimp ${OUTPUT_NPP}

profile_npp:
	ncu -o ${REPORT_NCU_NPP} \
	$(BUILD_DIR)/$(EXECUTABLE_NAME_NPP) ${INPUT} ${OUTPUT_NPP}

# kernel
run:
	$(BUILD_DIR)/$(EXECUTABLE_NAME_KERNEL)_${VERSION} ${INPUT} ${OUTPUT_KERNEL}_${VERSION}.pgm

see:
	gimp ${OUTPUT_KERNEL}_${VERSION}.pgm

profile:
	ncu -o ${REPORT_NCU_KERNEL}_${VERSION} -f \
	$(BUILD_DIR)/$(EXECUTABLE_NAME_KERNEL)_${VERSION} ${INPUT} ${OUTPUT_KERNEL}_${VERSION}.pgm

# clean up
clean:
	@rm -rf $(BUILD_DIR) output profile
