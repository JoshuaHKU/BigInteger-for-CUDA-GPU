
SRC = test_gpubigint.cu

OBJDIR = obj

OBJET = $(addprefix $(OBJDIR)/, \
      test_gpubigint.o)

CXX        = g++
CUDA       = /usr/local/cuda
CXXCUDA    = /usr/bin/g++-11
NVCC       = $(CUDA)/bin/nvcc


ifdef debug
CXXFLAGS   = -std=c++17 -DWITHGPU -m64  -mssse3 -Wno-unused-result -Wno-write-strings -g -I. -I$(CUDA)/include -I/usr/lib/include 
else
CXXFLAGS   = -std=c++17 -DWITHGPU -m64 -mssse3 -Wno-unused-result -Wno-write-strings -O2 -I. -I$(CUDA)/include -I/usr/lib/include 
endif
LFLAGS     = -lpthread -L$(CUDA)/lib64 -lcudart 
#当前GPU的架构及算力,这个要根据本机安装的Nvidia卡的类型指定
#RTX 3070对应的算力水平是compute_86 sm_86
#Tesla M40加速卡对应的算力水平是 compute_52 sm_52
GPU_ARCH = compute_86
GPU_CODE = sm_86

ifdef debug
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
	$(NVCC) -G -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -g -I$(CUDA)/include -gencode=arch=$(GPU_ARCH),code=$(GPU_CODE) -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
else
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
	$(NVCC) -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O2 -I$(CUDA)/include -gencode=arch=$(GPU_ARCH),code=$(GPU_CODE) -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
endif

$(OBJDIR)/%.o : %.cu
	$(NVCC) -ccbin $(CXXCUDA) -m64 -O2 -I. -I$(CUDA)/include -gencode=arch=$(GPU_ARCH),code=$(GPU_CODE) -c $< -o $@

all: test_gpubigint

test_gpubigint: $(OBJET)
	@echo Making gpu BigIntt test program...
	$(CXX) $(OBJET) $(LFLAGS)  -o test_gpubigint

$(OBJET): | $(OBJDIR)  

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	@echo Cleaning...
	@rm -f obj/*.o
