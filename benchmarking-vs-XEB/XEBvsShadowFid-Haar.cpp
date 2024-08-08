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

int main(int argc, char *argv[]){
    if (argc != 5) {
        cout << "./XEBvsShadowFid-Haar n N p seed" << endl;
        return 1;
    }

    int n = atoi(argv[1]);
    int d = (1 << n);
    int N = atoi(argv[2]);
    double p = atof(argv[3]);

    double normalization_pb = 0;
    vector<complex<double> > psi;
    vector<double> prob;
    for(int b = 0; b < (1 << n); b++){
        complex<double> psib = normal(gen) + rmi * normal(gen);
        psi.push_back(psib);

        double pb = abs(psib) * abs(psib);
        normalization_pb += pb;
    }
    normalization_pb = sqrt(normalization_pb);
    for(int b = 0; b < (1 << n); b++){
        psi[b] /= normalization_pb;
        prob.push_back(abs(psi[b]) * abs(psi[b]));
    }

    // seed is only for measurement
    gen = mt19937(atoi(argv[4]));

    double fid = (1 - p) + p / d;
    double local_fid = 0.0;
    double XEB = 0, normalization_XEB = 0;
    for(int b = 0; b < (1 << n); b++)
        normalization_XEB += prob[b] * prob[b];

    for(int rounds = 0; rounds < N; rounds++){
        double r = unif(gen), cur = 0.0;
        int outcome_b = -1;
        for(int b = 0; b < (1 << n); b++){
            if(r >= cur && r < cur + prob[b]){
                outcome_b = b;
                break;
            }
            cur += prob[b];
        }
        if(unif(gen) < p) // white noise
            outcome_b = int(unif(gen) * (1 << n));
        assert(outcome_b >= 0);
        double true_logp = log(prob[outcome_b]);
        XEB += exp(true_logp) / N;

        if(unif(gen) < p) // white noise
            local_fid += (1.0 * int(unif(gen) * 2)) / N;
        else
            local_fid += 1.0 / N;
    }

    double normalized_XEB = (XEB - pow(0.5, n)) / (normalization_XEB - pow(0.5, n));
    cout << "XEB: " << normalized_XEB << endl;
    cout << "ShadowFidelity: " << 1.0 * (d - 1) / d * 2 * (local_fid - 0.5) + (1.0 / d) << endl;
    cout << "Fidelity: " << fid << endl;

    return 0;
}
