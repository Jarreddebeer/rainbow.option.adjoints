#include <cstdlib>
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
    float2* d_ST_max, 
    float2* d_ST_del,
    float2* d_ST_veg, 
    int2* d_ST_aid,
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
    __shared__ float2 shared_mem[12*BLOCKSIZE];
    // assets:0->17, chlsky:18->26, Z_indp:28->(28+6*BLOCKSIZE), x23:(28+6*BLOCKSIZE)->(28+9*BLOCKSIZE), x24:(28+9*BLOCKSIZE)->(28+12*BLOCKSIZE)
    // note: Z_indp must be aligned in memory, this is why index '27' is skipped when assigning Z_indp


if (tid < num_sims) {

//    float* __restrict__ assets = &shared_mem[0]; // 3 * 6 [0-17]
//    float* __restrict__ chlsky = &shared_mem[18]; // 3 * 3 [18-26];
    float2* __restrict__ Z_indp = &shared_mem[6*threadIdx.x];
    float2* __restrict__ x23    = &shared_mem[6*BLOCKSIZE+3*threadIdx.x];
    float2* __restrict__ x24    = &shared_mem[9*BLOCKSIZE+3*threadIdx.x];

    //float2 Z_indp[3];
    float2  Z_corr[3];
    float assets[18];
    float chlsky[9];

    int2 winning_asset;
    float2 ST_max, ST_del, ST_veg;

    float2 x_8, x_7, x_6, x_5, x_4, x_3, x_2, x_1;
    float2 x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22;
    float2 _x24, _x22, _x19, _x15, _x9, _x0;
    float2 _x23, _x21, _x17, _x16, _x13, _x11, _x8, _x5v, _x0v;
    float2 _x_1[3], _x_2[3], _x_2v[3];
    float2 Zs, Zv;
    float2 payoff;

    // fetch asset and chlsky data
    for (int i=0; i<18; i++) { assets[i] = d_assets[i]; }
    for (int i=0; i<9; i++)  { chlsky[i] = d_chlsky[i]; }
//    if (threadIdx.x < 18) { shared_mem[threadIdx.x] = d_assets[threadIdx.x]; }
//    if (threadIdx.x < 9)  { shared_mem[18+threadIdx.x] = d_chlsky[threadIdx.x]; } 
    // done fetching
     
    // random number generator and state
    curandStatePhilox4_32_10 rng_state;
    curand_init(1234, tid, 0, &rng_state);

    winning_asset = make_int2(0, 0);
    ST_max = make_float2(0.0f, 0.0f);
    ST_del = make_float2(0.0f, 0.0f);
    ST_veg = make_float2(0.0f, 0.0f);

    for (int a=0; a<num_assets; a++) {

        x23[a] = make_float2(assets[1*num_assets+a],
                             assets[1*num_assets+a]); // vega  

        x24[a] = make_float2(assets[0*num_assets+a],
                             assets[0*num_assets+a]); // asset

        // init sensitivity vector
        _x_1[a] = make_float2(1.0f, 1.0f);
        _x_2[a] = make_float2(0.0f, 0.0f);
       _x_2v[a] = make_float2(1.0f, 1.0f); 

     }

     for (int t=0; t<num_steps; t++)
     {
           // generate correlated rngs
         for (int a=0; a<num_assets; a++)
             { reinterpret_cast<float4*>(&Z_indp[4*a])[0] = curand_normal4(&rng_state); Z_corr[a] = make_float2(0.0f, 0.0f); }
         for (int row=0; row<num_assets; row++) for (int c=0; c<num_assets; c++)
             { Z_corr[row] = make_float2(Z_corr[row].x + chlsky[num_assets*row+c] * Z_indp[2*c].x,
                                         Z_corr[row].y + chlsky[num_assets*row+c] * Z_indp[2*c].y); }

         for (int a=0; a<num_assets; a++)
         {
          
            // init input parameters
            x_8 = make_float2(assets[2*num_assets+a], assets[2*num_assets+a]); // rho
            x_7 = make_float2(assets[3*num_assets+a], assets[3*num_assets+a]); // kappa
            x_6 = make_float2(assets[4*num_assets+a], assets[4*num_assets+a]); // theta
            x_5 = make_float2(assets[5*num_assets+a], assets[5*num_assets+a]); // sigma
            x_4 = make_float2(dt, dt);
            x_3 = make_float2(r, r);
            x_2 = x23[a];
            x_1 = x24[a];

            Zs = Z_corr[a];
            Zv = make_float2(x_8.x * Zs.x + sqrt(1-x_8.x*x_8.x) * Z_indp[2*a+1].x,
                             x_8.y * Zs.y + sqrt(1-x_8.y*x_8.y) * Z_indp[2*a+1].y);

            // forward pass

            x0 = make_float2(x_2.x * x_4.x,
                             x_2.y * x_4.y);

            x1 = make_float2(x_3.x * x_4.x,
                             x_3.y * x_4.y);

            x2 = make_float2(x_6.x * x_4.x,
                             x_6.y * x_4.y);

            x3 = make_float2(x_5.x * x_5.x,
                             x_5.y * x_5.y);

            x4 = make_float2(Zv.x * Zv.x,
                             Zv.y * Zv.y);

            x5 = make_float2(sqrt(x0.x),
                             sqrt(x0.y));

            x6 = make_float2(x3.x * x_4.x,
                             x3.y * x_4.y);

            x7 = make_float2(x_7.x * x2.x,
                             x_7.y * x2.y);

            x8 = make_float2(x_7.x * x0.x,
                             x_7.y * x0.y);

            x9 = make_float2(-0.5f * x0.x,
                             -0.5f * x0.y);

            x10 = make_float2(x4.x - 1,
                              x4.y - 1);

            x11 = make_float2(x_5.x * x5.x,
                              x_5.y * x5.y);

            x12 = make_float2(Zs.x * x5.x,
                              Zs.y * x5.y);

            x13 = make_float2(x_2.x + x7.x,
                              x_2.y + x7.y);

            x14 = make_float2(0.25f * x10.x,
                              0.25f * x10.y);

            x15 = make_float2(x9.x + x12.x,
                              x9.y + x12.y);

            x16 = make_float2(Zv.x * x11.x,
                              Zv.y * x11.y);

            x17 = make_float2(x13.x - x8.x,
                              x13.y - x8.y);

            x18 = make_float2(x14.x * x6.x,
                              x14.y * x6.y);

            x19 = make_float2(x1.x + x15.x,
                              x1.y + x15.y);

            x20 = make_float2(x17.x + x16.x,
                              x17.y + x16.y);

            x21 = make_float2(x20.x + x18.x,
                              x20.y + x18.y);

            x22 = make_float2(exp(x19.x),
                              exp(x19.y));

            x23[a] = make_float2(max(x21.x, 0.0f),
                                 max(x21.y, 0.0f));

            x24[a] = make_float2(x_1.x * x22.x,
                                 x_1.y * x22.y);


            // adjoint pass

            // asset adjoints

            _x24 = make_float2(1, 1);
            _x22 = make_float2(x_1.x * _x24.x,
                               x_1.y * _x24.y);

            _x19 = make_float2(x22.x * _x22.x,
                               x22.y * _x22.y);

            _x15 = _x19;
            _x9 = _x15;
            _x0 = make_float2(-0.5f * _x9.x,
                              -0.5f * _x9.y);


            // volatility adjoints
            _x23 = make_float2(1.0f, 1.0f);
            _x21 = make_float2((x23[a].x > 0.0f) ? _x23.x : 0.0f,
                               (x23[a].y > 0.0f) ? _x23.y : 0.0f);

            _x13 = _x16 = _x17 = _x21;

            _x11 = make_float2(Zv.x * _x16.x,
                               Zv.y * _x16.y);

            _x8 = make_float2(-1 * _x17.x,
                              -1 * _x17.y);

            _x5v = make_float2(x_5.x * _x11.x,
                               x_5.y * _x11.y);

            _x0v = make_float2((0.5f * _x5v.x / x5.x) + _x8.x * x_7.x,
                               (0.5f * _x5v.y / x5.y) + _x8.y * x_7.y);


            // xbar inputs
            _x_1[a] = make_float2((_x24.x * x22.x) * _x_1[a].x,
                                  (_x24.y * x22.y) * _x_1[a].y);

            _x_2[a] = make_float2(_x_2[a].x * x22.x - (_x0.x *x_4.x) * _x_2v[a].x * (-1 + Zs.x / x5.x),
                                  _x_2[a].y * x22.y - (_x0.y *x_4.y) * _x_2v[a].y * (-1 + Zs.y / x5.y));

            _x_2v[a] = make_float2(_x_2v[a].x * (_x13.x + _x0v.x * x_4.x),
                                   _x_2v[a].y * (_x13.y + _x0v.y * x_4.y));


         }
     }

     for (int a=0; a<num_assets; a++) {

         payoff = make_float2(max(x24[a].x-K, 0.0f),
                              max(x24[a].y-K, 0.0f));

         if (payoff.x > ST_max.x) {
             winning_asset.x = a;
             ST_max.x = payoff.x;
             ST_del.x = _x_1[a].x;
             ST_veg.x = _x_2[a].x;
         }

         if (payoff.y > ST_max.y) {
             winning_asset.y = a;
             ST_max.y = payoff.y;
             ST_del.y = _x_1[a].y;
             ST_veg.y = _x_2[a].y;
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

    num_sims /= 2;

    cin >> num_assets;

    dim3 dimBlock(BLOCKSIZE, 1, 1);
    dim3 dimGrid(ceil( ((float)num_sims)/BLOCKSIZE ), 1, 1);
    int ST_bytes = num_sims * sizeof(float2);
    int aid_bytes = num_sims * sizeof(int2);
    int assets_bytes = num_assets * 6 * sizeof(float);
    int chlsky_bytes = num_assets * num_assets * sizeof(float);

    // allocate memory for gpu and host
    float2* h_ST_max = (float2*) malloc(ST_bytes);
    float2* h_ST_del = (float2*) malloc(ST_bytes);
    float2* h_ST_veg = (float2*) malloc(ST_bytes);
    int2*    h_ST_aid = (int2*)    malloc(aid_bytes);
    float2* d_ST_max = (float2*) malloc(ST_bytes);
    float2* d_ST_del = (float2*) malloc(ST_bytes);
    float2* d_ST_veg = (float2*) malloc(ST_bytes);
    int2*    d_ST_aid = (int2*)    malloc(aid_bytes);
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
        price += (h_ST_max[i].x + h_ST_max[i].y);
        delta[(int)h_ST_aid[i].x] += h_ST_del[i].x;
        delta[(int)h_ST_aid[i].y] += h_ST_del[i].y;
         vega[(int)h_ST_aid[i].x] += h_ST_veg[i].x;
         vega[(int)h_ST_aid[i].y] += h_ST_veg[i].y;
    }    
    double disc_fac = exp(-r*T);
    num_sims *= 2;
    price = disc_fac * price / num_sims;

    printf("--------------------------------------\n");
    printf("Heston 3 assets rainbow call on max\n");
    printf("price: %0.15g\n", price);
    for (int i=0; i<num_assets; i++) {
        printf("delta %d: %0.15g\n", disc_fac * delta[i] / num_sims);
        printf("vega %d: %0.15g\n",  disc_fac * vega[i]  / num_sims);
    }

    duration = second()-start-overhead;

    printf("======================================\n");
    printf("duration: %0.15g\n", duration);
    printf("======================================\n");

    return 0;
}
