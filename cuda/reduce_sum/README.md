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
nvprof --metrics gld_throughput ./cuda_executable
```

# 规约问题


