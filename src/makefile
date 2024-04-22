
SYS = mac
#SYS = lassen

FLAGS = -g -O3
DEFS = 
#DEFS += -D_MPI

INCS = -I. -I.. -I./force_types -I./mobility_types
LIBS = 

ifeq ($(SYS),mac)
	# KOKKOS OpenMP
	KOKKOS_PATH=/Users/bertin1/Documents/Codes/Libraries/kokkos/install_macos_clang
	KOKKOS_DEVICES=OpenMP
	INCS += -I$(KOKKOS_PATH)/include
	LIBS += -L$(KOKKOS_PATH)/lib -lkokkoscore -ldl
endif
ifeq ($(SYS),lassen)
	# KOKKOS Cuda
	KOKKOS_PATH=/usr/workspace/bertin1/KOKKOS/kokkos4.2.0/install_lassen_gcc_cuda
	KOKKOS_DEVICES=Cuda,OpenMP
	KOKKOS_ARCH=Volta70
	INCS += -I$(KOKKOS_PATH)/include
	LIBS += -L$(KOKKOS_PATH)/lib64 -lkokkoscore -ldl
endif

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
	CXX = $(KOKKOS_PATH)/bin/nvcc_wrapper
	FLAGS += -std=c++14 --expt-extended-lambda -arch=sm_70
else
	ifeq ($(SYS),mac)
		ifneq (,$(findstring D_MPI,$(DEFS)))
			CXX = /opt/local/bin/mpicxx
		else
			CXX = /opt/local/bin/clang++-mp-7.0
		endif
	else ifeq ($(SYS),lassen)
		CXX = /usr/tce/packages/gcc/gcc-8.3.1/bin/g++
	else
		ifneq (,$(findstring D_MPI,$(DEFS)))
			CXX = mpicxx
		else
			CXX = c++
		endif
	endif
	FLAGS += -std=c++14
endif

ifneq (,$(findstring OpenMP,$(KOKKOS_DEVICES)))
	FLAGS += -fopenmp
endif

TARGET = exadis

TARGET_SRCS = main.cpp \
              system.cpp \
			  generate.cpp \
			  utils.cpp \
			  network.cpp \
			  force_types/force_common.cpp
TARGET_OBJS = $(TARGET_SRCS:.cpp=.o)

%.o: %.cpp
	$(CXX) -c -o $@ $< $(FLAGS) $(DEFS) $(INCS)

all: $(TARGET)

$(TARGET): $(TARGET_OBJS)
	$(CXX) -o $@ $^ $(FLAGS) $(LIBS) $(LIBS.$(SYS))

clean:
	rm -f $(TARGET_OBJS) $(TARGET) *~ core *~ .DS_Store */.DS_Store
	rm -r output
