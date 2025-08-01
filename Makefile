.PHONY: all build run see profile run_npp see_npp profile_npp clean

# executable
BUILD_DIR = build
EXECUTABLE_NAME_KERNEL = image_conv_kernel
EXECUTABLE_NAME_NPP = image_conv_npp


# input/output
INPUT=input/pebble.pgm
OUTPUT_NPP=output/pebble_blurred_npp.pgm
OUTPUT_KERNEL=output/pebble_blurred_kernel.pgm

REPORT_NCU_KERNEL=profile/report_ncu_kernel
REPORT_NCU_NPP=profile/report_ncu_npp

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake ..
	@cd $(BUILD_DIR) && cmake --build .


# kernel run
run:
	$(BUILD_DIR)/$(EXECUTABLE_NAME_KERNEL) ${INPUT} ${OUTPUT_KERNEL}

see:
	gimp ${OUTPUT_KERNEL}

# npp run
run_npp:
	$(BUILD_DIR)/$(EXECUTABLE_NAME_NPP) ${INPUT} ${OUTPUT_NPP}

see_npp:
	gimp ${OUTPUT_NPP}


# profiling
profile:
	ncu -o ${REPORT_NCU_KERNEL} \
	$(BUILD_DIR)/$(EXECUTABLE_NAME_KERNEL) ${INPUT} ${OUTPUT_KERNEL}

profile_npp:
	ncu -o ${REPORT_NCU_NPP} \
	$(BUILD_DIR)/$(EXECUTABLE_NAME_NPP) ${INPUT} ${OUTPUT_NPP}


# clean up
clean:
	@rm -rf $(BUILD_DIR) output profile
