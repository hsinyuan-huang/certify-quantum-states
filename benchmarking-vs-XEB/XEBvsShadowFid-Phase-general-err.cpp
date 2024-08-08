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
        cout << "./XEBvsShadowFid-Phase-general-err n N p seed" << endl;
        return 1;
    }

    int n = atoi(argv[1]);
    int d = (1 << n);
    int N = atoi(argv[2]);
    double p = atof(argv[3]);

    vector<double> rotation;
    for(int i = 0; i < n; i ++)
        rotation.push_back((0.125 + 0.005 * normal(gen)) * 2 * M_PI);

    vector<complex<double> > psi;
    for(int b = 0; b < (1 << n); b++){
        complex<double> amplitude = 1.0;
        for(int i = 0; i < n; i++){
            int bi = (b & (1 << i)) >> i;
            if(bi == 0)
                amplitude *= cos(rotation[i]);
            else
                amplitude *= sin(rotation[i]);
        }

        double phase = unif(gen) * 2 * M_PI;
        complex<double> psib = amplitude * (cos(phase) + rmi * sin(phase));
        psi.push_back(psib);
    }
    vector<double> prob;
    for(int b = 0; b < (1 << n); b++){
        prob.push_back(abs(psi[b]) * abs(psi[b]));
    }

    complex<double> fid_c = 0;
    double normalization_pb_noisy = 0;
    vector<complex<double> > psi_noisy;
    vector<double> prob_noisy;

    for(int b = 0; b < (1 << n); b++){
        complex<double> psib = psi[b] + 0.75 * p * (normal(gen) + rmi * normal(gen)) / sqrt(d);
        double phase = 0.25 * p * normal(gen) * 2.0 * M_PI;
        psib *= (cos(phase) + rmi * sin(phase));
        psi_noisy.push_back(psib);

        double pb = abs(psib) * abs(psib);
        normalization_pb_noisy += pb;
    }
    normalization_pb_noisy = sqrt(normalization_pb_noisy);
    for(int b = 0; b < (1 << n); b++){
        psi_noisy[b] /= normalization_pb_noisy;
        prob_noisy.push_back(abs(psi_noisy[b]) * abs(psi_noisy[b]));
        fid_c += conj(psi_noisy[b]) * psi[b];
    }
    // double fid = (1.0-p) * (abs(fid_c) * abs(fid_c)) + p * pow(0.5, n);
    double fid = (abs(fid_c) * abs(fid_c));


    // seed is only for measurement
    gen = mt19937(atoi(argv[4]));

    double local_fid = 0.0;
    double XEB = 0, normalization_XEB = 0;
    for(int b = 0; b < (1 << n); b++)
        normalization_XEB += prob[b] * prob[b];

    for(int rounds = 0; rounds < N; rounds++){
        double r = unif(gen), cur = 0.0;
        int outcome_b = -1;
        for(int b = 0; b < (1 << n); b++){
            if(r >= cur && r < cur + prob_noisy[b]){
                outcome_b = b;
                break;
            }
            cur += prob_noisy[b];
        }
        // if(unif(gen) < p) // white noise
            // outcome_b = int(unif(gen) * (1 << n));
        assert(outcome_b >= 0);
        double true_logp = log(prob[outcome_b]);
        XEB += exp(true_logp) / N;

        // if(unif(gen) < p){ // white noise
            // local_fid += (1.0 * int(unif(gen) * 2)) / N;
        // }
        // else{
            int rand_i = int(unif(gen) * n);
            int b0 = outcome_b - (outcome_b & (1 << rand_i));
            int b1 = b0 + (1 << rand_i);
            complex<double> sfid_c = (conj(psi_noisy[b0]) * psi[b0] + conj(psi_noisy[b1]) * psi[b1]) / (sqrt(prob[b0] + prob[b1]) * sqrt(prob_noisy[b0] + prob_noisy[b1]));
            double sfid = abs(sfid_c) * abs(sfid_c);
            local_fid += (unif(gen) < sfid ? 1.0 : 0.0) / N;
        // }
    }

    double normalized_XEB = (XEB - pow(0.5, n)) / (normalization_XEB - pow(0.5, n));
    cout << "XEB: " << normalized_XEB << endl;
    cout << "ShadowFidelity: " << 1.0 * (d - 1) / d * 2 * (local_fid - 0.5) + (1.0 / d) << endl;
    cout << "Fidelity: " << fid << endl;

    return 0;
}
