BUILDNAME = nw
INCDIR = include
SRCDIR = src

ifeq ($(DEBUG),true)
	FLAGS = -lcurand -ccbin=g++ -std=c++11 -G -g -lineinfo -Wno-deprecated-gpu-targets
	OBJDIR := obj/debug
	BLDDIR := bin/debug
else
	FLAGS = -lcurand -ccbin=g++ -std=c++11 -Xptxas -O3,-v -Wno-deprecated-gpu-targets
	OBJDIR := obj/release
	BLDDIR := bin/release
endif

# Source files & object files files
CPP_FILES := $(wildcard $(SRCDIR)/*.cpp)
OBJ_FILES := $(addprefix $(OBJDIR)/,$(notdir $(CPP_FILES:.cpp=.o)))

# Creates dirs, makes external deps,
# compiles src files to obj then
# links into executable
build: dirs $(OBJ_FILES)
	nvcc $(FLAGS) $(OBJ_FILES) -I $(INCDIR) -o $(BLDDIR)/$(BUILDNAME) main.cu

# compiles src files into object files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	nvcc -x cu $(FLAGS) -I. -I $(INCDIR) -dc $< -o $@


dirs:
	@mkdir -p $(OBJDIR) $(BLDDIR)

clean:
	@rm -r $(OBJDIR)/*
	@rm -rf $(BLDDIR)/*

run:
	@./$(BLDDIR)/nw

test:
	echo $(CPP_FILES)