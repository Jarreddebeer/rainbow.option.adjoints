#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iomanip>
#include <vector>
#include "asset.h"
#include "second.h"

#define BLOCKSIZE 128

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

static __global__ void adjoint_method_correlation_GPU(
    float* d_ST_max, 
    float* d_ST_del,
    float* d_ST_veg, 
    int* d_ST_aid,
    float* d_assets,
    float* d_chlsky,
    int num_sims,
    int num_steps,
    int num_assets,
    float dt,
    float r,
    float K
) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

if (tid < num_sims) {

    float assets[18]; // 3 * 6
    float chlsky[9];  // 3 * 3
    float2 Z_indp[3];
    float  Z_corr[3];

    int winning_asset;
    float ST_max, ST_del, ST_veg;

    float x_8, x_7, x_6, x_5, x_4, x_3, x_2, x_1;
    float x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23[3], x24[3];
    float _x24, _x22, _x19, _x15, _x9, _x0;
    float _x23, _x21, _x17, _x16, _x13, _x11, _x8, _x5v, _x0v;
    float _x_1[3], _x_2[3], _x_2v[3];
    float Zs, Zv;
    float payoff;

    // fetch asset and chlsky data
    for (int i=0; i<18; i++) { assets[i] = d_assets[i]; }
    for (int i=0; i<9; i++)  { chlsky[i] = d_chlsky[i]; }
    // done fetching
     
    // random number generator and state
    curandStatePhilox4_32_10 rng_state;
    curand_init(1234, tid, 0, &rng_state);

    winning_asset = 0;
    ST_max = ST_del = ST_veg = 0.0;

    for (int a=0; a<num_assets; a++) {
        x23[a] = assets[1*num_assets+a]; // vega  
        x24[a] = assets[0*num_assets+a]; // asset
        // init sensitivity vector
        _x_1[a] = 1.0; _x_2[a] = 0.0; _x_2v[a] = 1.0; 
     }

     for (int t=0; t<num_steps; t++)
     {
           // generate correlated rngs
         for (int a=0; a<num_assets; a++) {
            double2 Z_indp_tmp = curand_normal2_double(&rng_state);
            Z_indp[a].x = (float)Z_indp_tmp.x;
            Z_indp[a].y = (float)Z_indp_tmp.y;
            Z_corr[a] = 0.0; }
         for (int row=0; row<num_assets; row++) for (int c=0; c<num_assets; c++) { Z_corr[row] += chlsky[num_assets*row+c] * Z_indp[c].x; }

         for (int a=0; a<num_assets; a++)
         {
          
            // init input parameters
            x_8 = assets[2*num_assets+a]; // rho
            x_7 = assets[3*num_assets+a]; // kappa
            x_6 = assets[4*num_assets+a]; // theta
            x_5 = assets[5*num_assets+a]; // sigma
            x_4 = dt;
            x_3 = r;
            x_2 = x23[a];
            x_1 = x24[a];

            Zs = Z_corr[a];
            Zv = x_8 * Zs + sqrt(1-x_8*x_8) * Z_indp[a].y;

            // forward pass

            x0 = x_2 * x_4;
            x1 = x_3 * x_4;
            x2 = x_6 * x_4;
            x3 = x_5 * x_5;
            x4 = Zv * Zv;
            x5 = sqrt(x0);
            x6 = x3 * x_4;
            x7 = x_7 * x2;
            x8 = x_7 * x0;
            x9 = -0.5 * x0;
            x10 = x4 - 1;
            x11 = x_5 * x5;
            x12 = Zs * x5;
            x13 = x_2 + x7;
            x14 = 0.25 * x10;
            x15 = x9 + x12;
            x16 = Zv * x11;
            x17 = x13 - x8;
            x18 = x14 * x6;
            x19 = x1 + x15;
            x20 = x17 + x16;
            x21 = x20 + x18;

            x22 = exp(x19);
            x23[a] = max(x21, 0.0);
            x24[a] = x_1 * x22;

            // adjoint pass

            // asset adjoints

            _x24 = 1;
            _x22 = x_1 * _x24;
            _x19 = x22 * _x22;
            _x15 = _x19;
            _x9 = _x15;
            _x0 = -0.5 * _x9;

            // volatility adjoints
            _x23 = 1.0;
            _x21 = (x23[a] > 0.0) ? _x23 : 0.0;
            _x13 = _x16 = _x17 = _x21;
            _x11 = Zv * _x16;
            _x8 = -1 * _x17;
            _x5v = x_5 * _x11;
            _x0v = (0.5 * _x5v / x5) + _x8*x_7;

            // xbar inputs
            _x_1[a] = (_x24*x22) * _x_1[a];
            _x_2[a] = _x_2[a]*x22 - (_x0*x_4) * _x_2v[a] * (-1 + Zs / x5);
            _x_2v[a] = _x_2v[a] * (_x13 + _x0v*x_4);

         }
     }

     for (int a=0; a<num_assets; a++) {
         payoff = max(x24[a]-K, 0.0);
         if (payoff > ST_max) {
             winning_asset = a;
             ST_max = payoff;
             ST_del = _x_1[a];
             ST_veg = _x_2[a];
         }
     }

     d_ST_max[tid] = ST_max;
     d_ST_del[tid] = ST_del;
     d_ST_veg[tid] = ST_veg;
     d_ST_aid[tid] = winning_asset;

} // end if (tid < num_sims)
}


int main(int argc, char **argv)
{

    int num_sims, num_steps, num_assets;
    float dt, r, K, T;
    double overhead, start, duration, price, delta[3], vega[3];

    num_sims  = strtod(argv[1], NULL);
    num_steps = strtod(argv[2], NULL);

    cin >> num_assets;

    dim3 dimBlock(BLOCKSIZE, 1, 1);
    dim3 dimGrid(ceil( ((float)num_sims)/BLOCKSIZE ), 1, 1);
    int ST_bytes = num_sims * sizeof(float);
    int aid_bytes = num_sims * sizeof(int);
    int assets_bytes = num_assets * 6 * sizeof(float);
    int chlsky_bytes = num_assets * num_assets * sizeof(float);

    // allocate memory for gpu and host
    float* h_ST_max = (float*) malloc(ST_bytes);
    float* h_ST_del = (float*) malloc(ST_bytes);
    float* h_ST_veg = (float*) malloc(ST_bytes);
    int*    h_ST_aid = (int*)    malloc(aid_bytes);
    float* d_ST_max = (float*) malloc(ST_bytes);
    float* d_ST_del = (float*) malloc(ST_bytes);
    float* d_ST_veg = (float*) malloc(ST_bytes);
    int*    d_ST_aid = (int*)    malloc(aid_bytes);
    //
    float* h_assets = (float*) malloc(assets_bytes);
    float* h_chlsky = (float*) malloc(chlsky_bytes);
    float* d_assets = (float*) malloc(assets_bytes);
    float* d_chlsky = (float*) malloc(chlsky_bytes);
    //
    gpuErrchk( cudaMalloc((void**) &d_ST_max, ST_bytes) );
    gpuErrchk( cudaMalloc((void**) &d_ST_del, ST_bytes) );
    gpuErrchk( cudaMalloc((void**) &d_ST_veg, ST_bytes) );
    gpuErrchk( cudaMalloc((void**) &d_ST_aid, aid_bytes) );
    gpuErrchk( cudaMalloc((void**) &d_assets, assets_bytes) );
    gpuErrchk( cudaMalloc((void**) &d_chlsky, chlsky_bytes) );

    // read asset parameters
    for (int i=0; i<num_assets; i++) {
        asset a;
        cin >> a.S >> a.V >> a.r >> a.T >> a.kappa >> a.theta >> a.sigma >> a.rho >> a.K;
        // these two are constant between assets (i'm being lazy here by re-assigning)
        T = a.T;
        dt = T / num_steps;
        r = a.r;
        K = a.K;
        //
        h_assets[0*num_assets+i] = a.S;
        h_assets[1*num_assets+i] = a.V;
        h_assets[2*num_assets+i] = a.rho;
        h_assets[3*num_assets+i] = a.kappa;
        h_assets[4*num_assets+i] = a.theta;
        h_assets[5*num_assets+i] = a.sigma;
    }

    // read lower cholesky decomposed matrix
    for (int i=0; i<num_assets; i++) {
        for (int j=0; j<num_assets; j++) {
            cin >> h_chlsky[num_assets*i+j];
        }
    }

    overhead = second()-second(); // overhead of timing method
    start = second();

    // copy data across
    gpuErrchk( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) );
    gpuErrchk( cudaMemcpy(d_assets, h_assets, assets_bytes, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_chlsky, h_chlsky, chlsky_bytes, cudaMemcpyHostToDevice) );

    adjoint_method_correlation_GPU<<<dimGrid, dimBlock>>>(
        d_ST_max, d_ST_del, d_ST_veg, d_ST_aid, d_assets, d_chlsky, num_sims, num_steps, num_assets, dt, r, K
    );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaMemcpy(h_ST_max, d_ST_max, ST_bytes, cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_ST_del, d_ST_del, ST_bytes, cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_ST_veg, d_ST_veg, ST_bytes, cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_ST_aid, d_ST_aid, aid_bytes, cudaMemcpyDeviceToHost) );


    for (int i=0; i<num_sims; i++) {
        price += h_ST_max[i];
        delta[(int)h_ST_aid[i]] += h_ST_del[i];
         vega[(int)h_ST_aid[i]] += h_ST_veg[i];
    }    
    double disc_fac = exp(-r*T);
    price = disc_fac * price / num_sims;

    printf("--------------------------------------\n");
    printf("Heston 3 assets rainbow call on max\n");
    printf("price 0: %0.15g\n", price);
    for (int i=0; i<num_assets; i++) {
        printf("delta %d: %0.15g\n", i, disc_fac * delta[i] / num_sims);
        printf("vega %d: %0.15g\n", i, disc_fac * vega[i]  / num_sims);
    }

    duration = second()-start-overhead;

    printf("======================================\n");
    printf("duration: %0.15g\n", duration);
    printf("======================================\n");

    return 0;
}
