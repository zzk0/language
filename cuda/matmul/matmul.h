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
