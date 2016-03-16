SHELL := /bin/bash
# Find out the base directory
BINDIR = bin
$(shell mkdir -p $(BINDIR))

.PHONY: default
default: all

# choose target
# Show list of all available targets for the current config with cfg_help
# Each config should at least provide the two targets RELEASE and DEBUG.
TARGET ?= RELEASE

.PHONY: help all gpuATTRACT emATTRACT mcATTRACT scATTRACT attract clean_as clean_em clean_mc clean_sc clean_attract cleanall_as cleanall_em cleanall_mc cleanall_sc cleanall_attract clean cleanall

em: as
	cd emATTRACT/bin && $(MAKE) TARGET=$(TARGET)
	cd $(BINDIR) && ln -sf ../emATTRACT/bin/emATTRACT
	
mc: as
	cd mcATTRACT/bin && $(MAKE) TARGET=$(TARGET)
	cd $(BINDIR) && ln -sf ../mcATTRACT/bin/mcATTRACT
	
sc: as
	cd scATTRACT/bin && $(MAKE) TARGET=$(TARGET)
	cd $(BINDIR) && ln -sf ../scATTRACT/bin/scATTRACT

as: check_cuda_version
	cd AttractServer/lib && $(MAKE) TARGET=$(TARGET)
	cd $(BINDIR) && ln -sf ../AttractServer/lib/libAttractServer.so

attract:
	cd attract/bin && $(MAKE) all

NVCC_RESULT = $(shell which nvcc 2> NULL)
NVCC_TEST = $(notdir $(NVCC_RESULT))

MIN_VERSION="6.5"
check_cuda_version:
	@echo "Makefile: Testing CUDA requirements"
	@if [ "$(NVCC_TEST)" == "nvcc" ]; then \
		echo -e >&2 "Makefile:\tOK: nvcc compiler found"; \
	else \
		echo -e >&2 "Makefile:\tERROR: nvcc compiler not found"; \
		false; \
	fi 
			
	@NVCC_VERSION=$$(nvcc --version | grep release | cut -d, -f2 | cut -d" " -f3); \
	if (( $$(echo "$$NVCC_VERSION >= $(MIN_VERSION)" | bc -l) )); then \
		echo -e >&2 "Makefile:\tOK: Compiler version $$NVCC_VERSION"; \
	else \
		echo -e >&2 "Makefile:\tERROR: Compiler version ($$NVCC_VERSION) too old. Upgrade to a version >= $(MIN_VERSION)"; \
		false; \
	fi

all: gpuATTRACT attract

gpuATTRACT: em mc sc

emATTRACT: em
mcATTRACT: mc
scATTRACT: sc

clean: clean_gpuATTRACT clean_attract
	
clean_as:
	cd AttractServer/lib && $(MAKE) clean
	
clean_em:
	cd emATTRACT/bin && $(MAKE) clean
	
clean_mc:
	cd mcATTRACT/bin && $(MAKE) clean
	
clean_sc:
	cd scATTRACT/bin && $(MAKE) clean

clean_attract:
	cd attract/bin && rm -f *.o

clean_gpuATTRACT: clean_as clean_em clean_mc clean_sc
 
cleanall: cleanall_as cleanall_em cleanall_mc cleanall_sc cleanall_attract
	rm -r $(BINDIR)
	
cleanall_as:
	cd AttractServer/lib && make cleanall
	
cleanall_em:
	cd emATTRACT/bin && make cleanall
	
cleanall_mc:
	cd mcATTRACT/bin && make cleanall
	
cleanall_sc:
	cd scATTRACT/bin && make cleanall

cleanall_attract:
	cd attract/bin && make clean

help:	
	@echo "Options:"
	@echo "make TARGET=DEBUG | RELEASE     choose target. (Default: RELEASE)"
	@echo
	@echo "e.g. make all TARGET=DEBUG -j4"
	@echo
	@echo "targets:"
	@echo "make all           build libAttractServer.so, emATTRACT, mcATTRACT, scATTRACT, attract"
	@echo "make gpuATTRACT    build libAttractServer.so, emATTRACT, mcATTRACT, scATTRACT
	@echo "make em            build libAttractServer.so and emATTRACT. (= make emATTRACT)"
	@echo "make mc            build libAttractServer.so and mcATTRACT. (= make mcATTRACT)"
	@echo "make sc            build libAttractServer.so and scATTRACT. (= make scATTRACT)"
	@echo "make attract       build original attract"  
	@echo "make clean         delete object and dependency files of all targets."
	@echo "make cleanall      delete object files, dependency files, executables, library and symbolic links of all targets."
	@echo "make clean_as      delete object and dependecy files of Attract-Server-library"
	@echo "make clean_em      delete object and dependecy files of emATTRACT."
	@echo "make clean_mc      delete object and dependecy files of mcATTRACT."
	@echo "make clean_sc      delete object and dependecy files of scATTRACT."
	@echo "make clean_attract delete object and dependecy files of attract."
	@echo "make cleanall_sc   delete object and dependency files, library of Attract-Server-library."
	@echo "make cleanall_em   delete object and dependency files, executable of emATTRACT."
	@echo "make cleanall_mc   delete object and dependency files, executable of mcATTRACT."
	@echo "make cleanall_sc   delete object and dependency files, executable of scATTRACT."
	@echo "make cleanall_attract delete object and dependency files, executable of attract."

	
	
