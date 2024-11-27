#include <cstdio>
#include <iostream>
#include <fstream>
#include <bitset>
#include <vector>
#include <random>
#include <complex>
#include <cmath>
#include <algorithm>

#include "circuit_simulation.h"

using namespace std;

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        cout << "./XEBvsShadowFid-Haar-dephasing n N p seed" << endl;
        return 1;
    }

    int n = atoi(argv[1]);
    int d = (1 << n);
    int N = atoi(argv[2]);
    double p = atof(argv[3]);

    double normalization_pb = 0;
    double collision_pb = 0;
    vector<complex<double>> psi;
    vector<double> prob;
    for (int b = 0; b < (1 << n); b++)
    {
        complex<double> psib = normal(gen) + rmi * normal(gen);
        psi.push_back(psib);

        double pb = abs(psib) * abs(psib);
        normalization_pb += pb;
    }
    normalization_pb = sqrt(normalization_pb);
    for (int b = 0; b < (1 << n); b++)
    {
        psi[b] /= normalization_pb;
        prob.push_back(abs(psi[b]) * abs(psi[b]));
        collision_pb += prob.back() * prob.back();
    }

    // seed is only for measurement
    gen = mt19937(atoi(argv[4]));

    double fid = (1 - p) + p * collision_pb;
    double local_fid = 0.0;
    double XEB = 0, normalization_XEB = 0;
    for (int b = 0; b < (1 << n); b++)
        normalization_XEB += prob[b] * prob[b];

    for (int rounds = 0; rounds < N; rounds++)
    {
        double r = unif(gen), cur = 0.0;
        int outcome_b = -1;
        for (int b = 0; b < (1 << n); b++)
        {
            if (r >= cur && r < cur + prob[b])
            {
                outcome_b = b;
                break;
            }
            cur += prob[b];
        }
        assert(outcome_b >= 0);
        double true_logp = log(prob[outcome_b]);
        XEB += exp(true_logp) / N;

        if (unif(gen) < p){ // phase noise
            int rand_i = int(unif(gen) * n);
            int b0 = outcome_b - (outcome_b & (1 << rand_i));
            int b1 = b0 + (1 << rand_i);

            double sfid = pow(prob[b0] / (prob[b0] + prob[b1]), 2.0) + pow(prob[b1] / (prob[b0] + prob[b1]), 2.0);
            local_fid += (unif(gen) < sfid ? 1.0 : 0.0) / N;
        }
        else
            local_fid += 1.0 / N;
    }

    double normalized_XEB = (XEB - pow(0.5, n)) / (normalization_XEB - pow(0.5, n));
    cout << "XEB: " << normalized_XEB << endl;
    cout << "ShadowFidelity: " << 3.0 * (local_fid - 2.0/3.0) * (1 - (2.0 / (d+1))) + (2.0 / (d+1)) << endl;
    cout << "Fidelity: " << fid << endl;

    return 0;
}
