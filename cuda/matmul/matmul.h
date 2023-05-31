void NaiveMatmul(float *A, float *B, float *C, int m, int n, int k);
void CublasMatmul(float *A, float *B, float *C, int m, int n, int k);

/**
分块计算，一个线程一个元素
*/
void BlockMatmul(float *A, float *B, float *C, int m, int n, int k);

/**
分块 + 一个线程计算多个元素
*/
void BlockWithStrideMatmul(float *A, float *B, float *C, int m, int n, int k);

/**
分块 + 一个线程计算多个元素 + 共享内存对齐
*/
void BlockWithStrideAlignMatmul(float *A, float *B, float *C, int m, int n, int k);

/**
分块 + 一个线程计算多个元素 + 重排 + 共享内存对齐
*/
void BlockWithTileReorderAlignMatmul(float *A, float *B, float *C, int m, int n, int k);

// /**
// 分块 + 一个线程计算多个元素 + 重排 + 共享内存对齐

// 优化每个 kernel 的寄存器使用，限制在 255 以内，避免使用栈空间
// */
// void BlockWithTileReorderAlignMatmul(float *A, float *B, float *C, int m, int n, int k);
