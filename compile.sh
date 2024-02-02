nvcc run.cu -o run_cute_32_4 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 -lnvToolsExt -lcusparse -Xcompiler -fopenmp -DTM=32 -DTN=4 -DERROR_CHECK=0 -DMachine=2
nvcc run.cu -o run_cute_16_4 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 -lnvToolsExt -lcusparse -Xcompiler -fopenmp -DTM=16 -DTN=4 -DERROR_CHECK=0 -DMachine=2
