# CUDA C 编程权威指南第三章

## GPU 架构

### 执行模型

GPU 上有多个流式多处理器 SM，一个 SM 包含一套完整的组件；启动 kernel 的时候，会以网格的形式配置启动，一个网格有多个线程块，线程块会被分发到 SM 上执行直到完成，多个线程块可以被分发到同一个 SM。线程块在 SM 上执行的时候，会被划分为 32 个线程的线程束，线程束的执行是以 SIMT 来执行的，一条指令，多个线程同时执行。SM 上的线程调度器会选择线程束，并且把一个指令发送给线程束执行。

有如下对应关系：

```
线程   <------> CUDA 核心
线程块 <------> SM
网格   <------> GPU
```

### 关键组件

- CUDA 核心
- 共享内存/一级缓存
- 寄存器文件
- 加载/存储单元
- 特殊功能单元（正弦、余弦、平方根、插值）
- 线程束调度器


## 工具

文中提到了两个工具：nvvp，独立的可视化分析器；nvprof，命令行分析器。

nvprof 常用的选项有：

```bash
# 分支效率：(分支数 - 分化分支数 / 分支数)
nvprof --metrics branch_efficiency ./cuda_executable
# 分支和分化分支的事件计数器
nvprof --metrics branch,divergent_branch ./cuda_executable
# 占有率：每周期内活跃线程束的平均数量与一个 SM 支持的线程束最大数量的比值
nvprof --metrics achieved_occupancy ./cuda_executable
# 内核的内存读取效率
nvprof --metrics gld_throughput ./cuda_executable
# 全局加载效率：被请求的全局加载吞吐量占所需的全局加载吞吐量的比值
nvprof --metrics gld_efficiency ./cuda_executable
# 每个线程束上执行指令数量的平均值
nvprof --metrics inst_per_warp ./cuda_executable
# 设备内存读取吞吐量
nvprof --metrics dram_read_throughput ./cuda_executable
```

# 规约问题

规约求和问题的实现方式：

- 朴素实现：每个元素依次相加
- 相邻配对：元素和它们直接相邻的元素配对相加
- 交错配对：根据跨度配对元素

优化的结果：

```
CPU reduce sum time cost: 12655 us
==1711099== NVPROF is profiling process 1711099, command: ./cuda/reduce_sum/reduce_sum
GPU warmup time cost: 50 us
time cost: 628 us       reduce neighbor 
time cost: 352 us       reduce neighbor + less divergence 
time cost: 316 us       reduce interleave 
time cost: 267 us       reduce interleave + loop unrolling x 2 
time cost: 170 us       reduce interleave + loop unrolling x 8 
time cost: 170 us       reduce interleave + loop unrolling x 8 + warp unrolling 
time cost: 169 us       reduce interleave + loop unrolling x 8 + warp unrolling + complete unroll 
time cost: 166 us       reduce interleave + loop unrolling x 8 + warp unrolling + complete unroll + template 
==1711099== Profiling application: ./cuda/reduce_sum/reduce_sum
==1711099== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   98.50%  127.90ms        10  12.790ms  1.8560us  14.657ms  [CUDA memcpy HtoD]
                    0.45%  585.22us         2  292.61us  4.0960us  581.12us  void GpuReduceSum0<int>(int*, int*, unsigned int)
                    0.24%  307.52us         1  307.52us  307.52us  307.52us  void GpuReduceSum1<int>(int*, int*, unsigned int)
                    0.21%  271.74us         1  271.74us  271.74us  271.74us  void GpuReduceSum2<int>(int*, int*, unsigned int)
                    0.17%  221.18us         1  221.18us  221.18us  221.18us  void GpuReduceSum3<int>(int*, int*, unsigned int)
                    0.10%  128.67us         1  128.67us  128.67us  128.67us  void GpuReduceSum6<int>(int*, int*, unsigned int)
                    0.10%  127.71us         1  127.71us  127.71us  127.71us  void GpuReduceSum5<int>(int*, int*, unsigned int)
                    0.10%  127.39us         1  127.39us  127.39us  127.39us  void GpuReduceSum4<int>(int*, int*, unsigned int)
                    0.10%  125.98us         1  125.98us  125.98us  125.98us  void GpuReduceSum7<int, unsigned int=512>(int*, int*, unsigned int)
                    0.04%  49.632us         8  6.2040us  2.4640us  11.136us  [CUDA memcpy DtoH]
      API calls:   61.89%  231.78ms        18  12.877ms  256.42us  220.96ms  cudaMalloc
                   36.09%  135.17ms        18  7.5095ms  29.872us  15.216ms  cudaMemcpy
                    0.76%  2.8330ms        18  157.39us  120.76us  186.48us  cudaFree
                    0.52%  1.9302ms        10  193.02us  6.1160us  584.49us  cudaDeviceSynchronize
                    0.46%  1.7237ms         2  861.83us  812.19us  911.48us  cuDeviceTotalMem
                    0.17%  654.52us       194  3.3730us     159ns  173.86us  cuDeviceGetAttribute
                    0.09%  331.95us         9  36.883us  33.211us  40.408us  cudaLaunchKernel
                    0.02%  70.878us         2  35.439us  18.809us  52.069us  cuDeviceGetName
                    0.00%  15.386us         2  7.6930us  2.7710us  12.615us  cuDeviceGetPCIBusId
                    0.00%  2.8350us         4     708ns     189ns  2.1100us  cuDeviceGet
                    0.00%  1.5550us         2     777ns     446ns  1.1090us  cuDeviceGetUuid
                    0.00%  1.2330us         3     411ns     189ns     682ns  cuDeviceGetCount
```

# 备注

当开启以下选项的时候，会出现 illegal memory access，原因未知。

```
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --relocatable-device-code=true
                       >)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
```

```
CPU reduce sum time cost: 14449 us
GPU warmup time cost: 31 us
time cost: 1750 us      reduce neighbor 
time cost: 955 us       reduce neighbor + less divergence 
time cost: 961 us       reduce interleave 
time cost: 51101 us     reduce sum recursive 
ERROR: /home/percent1/code/cmake_cpp_cuda/cuda/./include/smart_pointer.h:25,code:700,reason:an illegal memory access was encountered
```
