SHELL := /bin/bash
MAKEFILE_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
OS := $(shell uname -s)

# Allowed CPU values: k8, armv7a, aarch64
ifeq ($(OS),Linux)
CPU ?= k8
else
$(error $(OS) is not supported)
endif

ifeq ($(filter $(CPU),k8 armv7a aarch64),)
$(error CPU must be k8, armv7a, aarch64)
endif

# Allowed COMPILATION_MODE values: opt, dbg, fastbuild
COMPILATION_MODE ?= opt
ifeq ($(filter $(COMPILATION_MODE),opt dbg fastbuild),)
$(error COMPILATION_MODE must be opt, dbg, or fastbuild)
endif

BAZEL_OUT_DIR :=  $(MAKEFILE_DIR)/bazel-out/$(CPU)-$(COMPILATION_MODE)/bin
BAZEL_BUILD_FLAGS := --compilation_mode=$(COMPILATION_MODE) \
                     --cpu=$(CPU)

ifeq ($(CPU),aarch64)
BAZEL_BUILD_FLAGS += --copt=-ffp-contract=off
else ifeq ($(CPU),armv7a)
BAZEL_BUILD_FLAGS += --copt=-ffp-contract=off
endif

# $(1): pattern, $(2) destination directory
define copy_out_files
pushd $(BAZEL_OUT_DIR); \
for f in `find . -name $(1) -type f`; do \
	mkdir -p $(2)/`dirname $$f`; \
	cp -f $(BAZEL_OUT_DIR)/$$f $(2)/$$f; \
done; \
popd
endef

EXAMPLES_OUT_DIR   := $(MAKEFILE_DIR)/out/$(CPU)/examples

.PHONY: all \
        clean \
        help

all:

	bazel build $(BAZEL_BUILD_FLAGS) //coral/examples:classify_image \
	                                 //coral/examples:main
	mkdir -p $(EXAMPLES_OUT_DIR)
	cp -f $(BAZEL_OUT_DIR)/coral/examples/libclassify_image.so \
	      $(BAZEL_OUT_DIR)/coral/examples/main \
	      $(EXAMPLES_OUT_DIR)

clean:
	rm -rf $(MAKEFILE_DIR)/bazel-* \
	       $(MAKEFILE_DIR)/out

DOCKER_WORKSPACE := $(MAKEFILE_DIR)/$(if $(TEST_ENV),..,)
DOCKER_WORKSPACE_CD := $(if $(TEST_ENV),libcoral,)
DOCKER_CPUS := k8 armv7a aarch64
DOCKER_TAG_BASE := coral-edgetpu
include $(MAKEFILE_DIR)/docker/docker.mk

help:
	@echo "make all        - Build all C++ code"
	@echo "make clean      - Remove generated files"
	@echo "make help       - Print help message"
