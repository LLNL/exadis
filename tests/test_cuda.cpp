/*---------------------------------------------------------------------------
 *
 *  ExaDiS
 *
 *  Nicolas Bertin
 *  bertin1@llnl.gov
 *
 *-------------------------------------------------------------------------*/

#include <stdio.h>

#ifdef __CUDACC__

/*---------------------------------------------------------------------------
 *
 *    Function:     kernel
 *
 *-------------------------------------------------------------------------*/
__global__ void kernel(int N, float a, float *x, float *y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) y[i] = a*x[i];
}

/*---------------------------------------------------------------------------
 *
 *    Function:     main
 *
 *-------------------------------------------------------------------------*/
int main(void)
{
    int N = 1<<20;
    float *x, *y, *d_x, *d_y;
    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));

    cudaMalloc(&d_x, N*sizeof(float)); 
    cudaMalloc(&d_y, N*sizeof(float));

    for (int i = 0; i < N; i++)
        x[i] = 1.0f;

    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);

    kernel<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-2.0f));
    
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
    
    if (maxError < 1e-12) printf("PASS\n");
    else printf("FAIL\n");
    
    return 0;
}

#else

/*---------------------------------------------------------------------------
 *
 *    Function:     main
 *
 *-------------------------------------------------------------------------*/
int main(void) { 
    return 0;
}

#endif
