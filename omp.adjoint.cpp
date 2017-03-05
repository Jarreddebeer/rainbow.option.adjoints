#include <cstdlib>
#include <iostream>
#include <cmath>
#include <random>
#include <iomanip>
#include <vector>
#include <omp.h>
#include "asset.h"
#include "second.h"

using namespace std;

void adjoint_method_correlation(
    int num_sims,
    int num_steps,
    int num_assets,
    const vector<asset>& assets,
    double dt,
    const vector<vector<double> >& chol_decomp
) {

    double ST_max_sum;
    double ST_del_sum1, ST_del_sum2, ST_del_sum3;
    double ST_veg_sum1, ST_veg_sum2, ST_veg_sum3;

omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel reduction(+:ST_max_sum,ST_del_sum1,ST_del_sum2,ST_del_sum3,ST_veg_sum1,ST_veg_sum2,ST_veg_sum3)
{
 
    double x_7, x_6, x_5, x_4, x_3, x_2, x_1;
    double x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24;
    double _x24, _x22, _x19, _x15, _x12, _x9, _x5, _x1, _x0;
    double _x23, _x21, _x20, _x17, _x16, _x13, _x11, _x8, _x5v, _x0v, _x_2v;
    double _x_1, _x_2, _x_3;
    double y1, y2, Zs, Zv;

    vector<double> Z_indp(num_assets);
    vector<double> Z_corr(num_assets);
    vector<vector<double> > Z_corr_generated(num_assets, vector<double>(num_steps));
    double ST_max, ST_del, ST_veg;
    int winning_asset;

    vector<double> ST_del_store = vector<double>(num_assets);
    vector<double> ST_veg_store = vector<double>(num_assets);
    for (int a=0; a<num_assets; a++) { ST_del_store[a] = 0.0; ST_veg_store[a] = 0.0; }

    std::mt19937 generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    #pragma omp for
    for (int i=0; i<num_sims; i++) {

        for (int t=0; t<num_steps; t++) {
            // generate randoms for each asset
            for (int a=0; a<num_assets; a++) { Z_indp[a] = distribution(generator); }
            // correlate the generated randoms
            for (int a=0; a<num_assets; a++) Z_corr[a] = 0;
            for (int r=0; r<num_assets; r++) for (int c=0; c<num_assets; c++) {
                Z_corr[r] += chol_decomp[r][c] * Z_indp[c];
            }
            for (int a=0; a<num_assets; a++) Z_corr_generated[a][t] = Z_corr[a];
        }

        ST_max = ST_del = ST_veg = 0.0;
        winning_asset = 0;
        for (int a=0; a<num_assets; a++) {

            double rho = assets[a].rho;
            double kappa = assets[a].kappa;
            double theta = assets[a].theta;
            double sigma = assets[a].sigma;
            double r = assets[a].r;
            x24 = assets[a].S;
            x23 = assets[a].V;
            _x_1 = 1.0;
            _x_2 = 0.0;
            _x_2v = 1.0;

            for (int t=0; t<num_steps; t++)
            {

                // forward pass

                Zs = Z_corr_generated[a][t];
                Zv = rho * Zs + sqrt(1-rho*rho) * distribution(generator);

                x_7 = kappa;
                x_6 = theta;
                x_5 = sigma;
                x_4 = dt;
                x_3 = r;
                x_2 = x23;
                x_1 = x24;

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
                x23 = max(x21, 0.0);
                x24 = x_1 * x22;

                y1 = x24;
                y2 = x23;

                // adjoint pass

                // asset adjoints

                _x24 = 1;
                _x22 = x_1 * _x24;
                _x19 = x22 * _x22;
                _x15 = _x19;
                _x12 = _x15;
                _x9 = _x15;
                _x5 = Zs * _x12;
                _x1 = _x19;
                _x0 = -0.5 * _x9;

                // volatility adjoints
                _x23 = 1.0;
                _x21 = (x23 > 0.0) ? _x23 : 0.0;
                _x13 = _x16 = _x17 = _x20 = _x21 = 1;
                _x11 = Zv * _x16;
                _x8 = -1 * _x17;
                _x5v = x_5 * _x11;
                _x0v = (0.5 * _x5v / x5) + _x8*x_7;

                // xbar inputs
                _x_1 = (_x24*x22) * _x_1;
                _x_2 = _x_2*x22 - (_x0*x_4) * _x_2v * (-1 + Zs / x5);
                _x_2v = _x_2v * (_x13 + _x0v*x_4);


            }
            double payoff = max(y1-assets[a].K, 0.0);
            if (payoff > ST_max) {
                winning_asset = a;
                ST_max = payoff;
                ST_del = _x_1;
                ST_veg = _x_2;
            }
        }
        ST_max_sum += ST_max;
        if      (winning_asset == 0) { ST_del_sum1 += ST_del; ST_veg_sum1 += ST_veg; }
        else if (winning_asset == 1) { ST_del_sum2 += ST_del; ST_veg_sum2 += ST_veg;}
        else if (winning_asset == 2) { ST_del_sum3 += ST_del; ST_veg_sum3 += ST_veg;}
    }
} // OMP end parallel

    // calculate the rainbow option price
    // it is assumed r and T are the same for all assets. We simply take the 0th one.
    double disc_fac = exp(-assets[0].r * assets[0].T);
    double price = disc_fac * (ST_max_sum / num_sims);
    printf("--------------------------------------\n");
    printf("Heston 3 assets rainbow call on max\n");
    printf("price: %0.15g\n", price);
    //
    printf("delta 1%d: %0.15g\n", 1, disc_fac * (ST_del_sum1 / num_sims));
    printf("vega  1%d: %0.15g\n", 1, disc_fac * (ST_veg_sum1 / num_sims));
    printf("delta 2%d: %0.15g\n", 2, disc_fac * (ST_del_sum2 / num_sims));
    printf("vega  2%d: %0.15g\n", 2, disc_fac * (ST_veg_sum2 / num_sims));
    printf("delta 3%d: %0.15g\n", 3, disc_fac * (ST_del_sum3 / num_sims));
    printf("vega  3%d: %0.15g\n", 3, disc_fac * (ST_veg_sum3 / num_sims));

}


int main(int argc, char **argv)
{
    int num_assets, simSize, stepSize;
    double overhead, start, duration, dt;

    int num_sims  = strtod(argv[1], NULL);
    int num_steps = strtod(argv[2], NULL);

    cin >> num_assets;
    vector<asset> assets(num_assets);
    vector<vector<double> > chol_decomp(num_assets, vector<double>(num_assets));

    // read asset parameters
    for (int i=0; i<num_assets; i++) {
        asset a;
        cin >> a.S >> a.V >> a.r >> a.T >> a.kappa >> a.theta >> a.sigma >> a.rho >> a.K;
        assets[i] = a;
    }

    // read lower cholesky decomposed matrix
    for (int i=0; i<num_assets; i++) {
        for (int j=0; j<num_assets; j++) {
            cin >> chol_decomp[i][j];
        }
    }

    overhead = second()-second(); // overhead of timing method
    dt = assets[0].T / num_steps;

    start = second();
    adjoint_method_correlation(num_sims, num_steps, num_assets, assets, dt, chol_decomp);
    duration = second()-start-overhead;

    printf("======================================\n");
    printf("duration: %0.15g\n", duration);
    printf("======================================\n");

    return 0;
}
