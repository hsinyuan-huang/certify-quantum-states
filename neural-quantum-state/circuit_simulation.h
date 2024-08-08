#include <cstdio>
#include <iostream>
#include <fstream>
#include <bitset>
#include <vector>
#include <random>
#include <complex>
#include <cmath>
#include <algorithm>
#include <armadillo>

using namespace std;

// Random number generator
random_device rd;
mt19937 gen(13179);
uniform_real_distribution<double> unif(0.0, 1.0);
normal_distribution<double> normal(0.0, 1.0);

vector<double> readDoublesFromFile(const string& filename) {
    ifstream inputFile(filename);

    vector<double> betas;

    if (!inputFile.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return betas;
    }

    double value;
    while (inputFile >> value) {
        betas.push_back(value);
    }

    inputFile.close();

    return betas;
}

double sqrt_2 = sqrt(2);
double isq2 = 1./sqrt(2);
const std::complex<double> rmi(0.0, 1.0);

void Hadamard_transform(vector<complex<double> > &psi, int n){
    int d = (1 << n);
    for (int qbit = 0; qbit < n; qbit++) {
        vector<complex<double> > psi_tmp;
        for (int i = 0; i < d; ++i) { psi_tmp.push_back(0.0 + rmi * 0.0); }
        for (int j = 0; j < d; ++j) {
            int bit_parity=(j>>qbit)%2 ;
            if (bit_parity == 0) {
                psi_tmp[j] += isq2 * psi[j];
                int set_bit = j | (1<<qbit);
                psi_tmp[set_bit] += isq2 * psi[j];
            }
            else if (bit_parity == 1) {
                psi_tmp[j] += -isq2 * psi[j];
                int clear_bit = j & ~(1<<qbit);
                psi_tmp[clear_bit] += isq2 * psi[j];
            }
        }
        for (int i = 0; i < d; ++i) { psi[i] = psi_tmp[i]; }
    }
}

void print_psi(vector<complex<double> > psi, int n){
    printf("Psi:");
    for(int b = 0; b < (1 << n); b++){
        if(b > 0) printf("+");
        printf(" %.2f+%.2fi|%d> ", real(psi[b]), imag(psi[b]), b);
    }
    printf("\n");
}

vector<complex<double> > init_psi(int n){
    vector<complex<double> > psi;
    psi.push_back(1);
    for(int b = 1; b < (1 << n); b++){
        psi.push_back(0);
    }
    return psi;
}

void Hgate(vector<complex<double> > &psi, int n, int i){
    vector<complex<double> > Hpsi = psi;
    for(int b = 0; b < (1 << n); b++){
        int b0 = b - (b & (1 << i));
        int b1 = b | (1 << i);
        Hpsi[b0] = (psi[b0] + psi[b1]) / sqrt_2;
        Hpsi[b1] = (psi[b0] - psi[b1]) / sqrt_2;
    }
    psi = Hpsi;
}

void Xgate(vector<complex<double> > &psi, int n, int i){
    vector<complex<double> > Xpsi = psi;
    for(int b = 0; b < (1 << n); b++){
        int Xi_b = b ^ (1 << i);
        Xpsi[b] = psi[Xi_b];
        Xpsi[Xi_b] = psi[b];
    }
    psi = Xpsi;
}

void Bgate(vector<complex<double> > &psi, int n, int i, double beta){
    vector<complex<double> > Bpsi = psi;
    for(int b = 0; b < (1 << n); b++){
        int zero_or_one = ((b & (1 << i)) >> i);
        if(zero_or_one == 1)
            Bpsi[b] = psi[b] * exp(rmi * beta);
        else
            Bpsi[b] = psi[b];
    }
    psi = Bpsi;
}

void CZgate(vector<complex<double> > &psi, int n, int i, int j){
    vector<complex<double> > CZpsi = psi;
    for(int b = 0; b < (1 << n); b++){
        int ibit = ((b & (1 << i)) >> i);
        int jbit = ((b & (1 << j)) >> j);

        if(ibit == 1 && jbit == 1)
            CZpsi[b] = -psi[b];
        else
            CZpsi[b] = psi[b];
    }
    psi = CZpsi;
}

void Measurement(vector<complex<double> > &psi, int n, int i, int &outcome, double &logprob_outcome){
    vector<complex<double> > postM_psi = psi;

    double prob[2] = {0, 0};
    for(int b = 0; b < (1 << n); b++){
        int ibit = ((b & (1 << i)) >> i);
        prob[ibit] += abs(psi[b]) * abs(psi[b]);
    }

    double r = unif(gen);
    outcome = (r < prob[0]) ? 0 : 1;
    logprob_outcome = log(prob[outcome]);
    double sqrt_prob = sqrt(prob[outcome]);

    for(int b = 0; b < (1 << n); b++){
        int ibit = ((b & (1 << i)) >> i);

        if(ibit == outcome)
            postM_psi[b] = psi[b] / sqrt_prob;
        else
            postM_psi[b] = 0;
    }
    psi = postM_psi;
}

void Postselected_Measurement(vector<complex<double> > &psi, int n, int i, int outcome, double &logprob_outcome){
    vector<complex<double> > postM_psi = psi;

    double prob[2] = {0, 0};
    for(int b = 0; b < (1 << n); b++){
        int ibit = ((b & (1 << i)) >> i);
        prob[ibit] += abs(psi[b]) * abs(psi[b]);
    }

    logprob_outcome = log(prob[outcome]);
    double sqrt_prob = sqrt(prob[outcome]);

    for(int b = 0; b < (1 << n); b++){
        int ibit = ((b & (1 << i)) >> i);

        if(ibit == outcome)
            postM_psi[b] = psi[b] / sqrt_prob;
        else
            postM_psi[b] = 0;
    }
    psi = postM_psi;
}

double simulate_one_shot_MBQC(int n, int m, vector<int> input_state, vector<double> betas, vector<int> &measurement_outcome){
    vector<complex<double> > psi = init_psi(n);

    measurement_outcome = input_state;
    double log_prob_ttl = 0.0;

    for(int i = 0; i < n; i ++){
        for(int j = 0; j < m; j++){
            if(input_state[i*m + j] == 0){
                Hgate(psi, n, j);
                if(i > 0 && input_state[(i-1)*m + j] == 0 && measurement_outcome[(i-1)*m + j] == 1)
                    Xgate(psi, n, j);
                Bgate(psi, n, j, betas[i*m + j]);
            }
        }

        for(int j = 0; j < m; j+=2){
            if(j+1 >= m) continue;
            if(input_state[i*m+j] == 0 && input_state[i*m+j+1] == 0)
                CZgate(psi, n, j, j+1);
        }
        for(int j = 1; j < m; j+=2){
            if(j+1 >= m) continue;
            if(input_state[i*m+j] == 0 && input_state[i*m+j+1] == 0)
                CZgate(psi, n, j, j+1);
        }

        for(int j = 0; j < m; j++){
            if(input_state[i*m + j] == 0){
                if(i < n-1 && input_state[(i+1)*m + j] == 0){
                    double r = unif(gen);
                    measurement_outcome[i*m+j] = (r < 0.5) ? 0 : 1;
                    log_prob_ttl += log(0.5);
                }
                else{
                    Hgate(psi, n, j);

                    int outcome;
                    double logprob_outcome;
                    Measurement(psi, n, j, outcome, logprob_outcome);
                    measurement_outcome[i*m+j] = outcome;
                    log_prob_ttl += logprob_outcome;

                    if(outcome == 1)
                        Xgate(psi, n, j);
                }
            }
            if(input_state[i*m + j] == 1){
                double r = unif(gen);
                measurement_outcome[i*m+j] = (r < 0.5) ? 0 : 1;
                log_prob_ttl += log(0.5);
            }
        }
    }

    return log_prob_ttl;
}


double evaluate_one_shot_MBQC(int n, int m, vector<int> input_state, vector<int> measurement_outcome, vector<double> betas){
    vector<complex<double> > psi = init_psi(n);

    double log_prob_ttl = 0.0;

    for(int i = 0; i < n; i ++){
        for(int j = 0; j < m; j++){
            if(input_state[i*m + j] == 0){
                Hgate(psi, n, j);
                if(i > 0 && input_state[(i-1)*m + j] == 0 && measurement_outcome[(i-1)*m + j] == 1)
                    Xgate(psi, n, j);
                Bgate(psi, n, j, betas[i*m + j]);
            }
        }

        for(int j = 0; j < m; j+=2){
            if(j+1 >= m) continue;
            if(input_state[i*m+j] == 0 && input_state[i*m+j+1] == 0)
                CZgate(psi, n, j, j+1);
        }
        for(int j = 1; j < m; j+=2){
            if(j+1 >= m) continue;
            if(input_state[i*m+j] == 0 && input_state[i*m+j+1] == 0)
                CZgate(psi, n, j, j+1);
        }

        for(int j = 0; j < m; j++){
            if(input_state[i*m + j] == 0){
                if(i < n-1 && input_state[(i+1)*m + j] == 0)
                    log_prob_ttl += log(0.5);
                else{
                    Hgate(psi, n, j);

                    int outcome = measurement_outcome[i*m+j];
                    double logprob_outcome;
                    Postselected_Measurement(psi, n, j, outcome, logprob_outcome);
                    log_prob_ttl += logprob_outcome;

                    if(outcome == 1)
                        Xgate(psi, n, j);
                }
            }
            if(input_state[i*m + j] == 1)
                log_prob_ttl += log(0.5);
        }
    }

    return log_prob_ttl;
}


arma::cx_mat init_Pauli(int n, int i, int P){
    arma::cx_mat Pauli(1 << n, 1 << n, arma::fill::zeros);

    for(int b = 0; b < (1 << n); b++){
        int b0 = b - (b & (1 << i));
        int b1 = b | (1 << i);

        if(P == 0){ // I
            Pauli(b0, b0) = 1;
            Pauli(b1, b1) = 1;
            Pauli(b0, b1) = 0;
            Pauli(b1, b0) = 0;
        }
        if(P == 1){ // X
            Pauli(b0, b0) = 0;
            Pauli(b1, b1) = 0;
            Pauli(b0, b1) = 1;
            Pauli(b1, b0) = 1;
        }
        if(P == 2){ // Y
            Pauli(b0, b0) = 0;
            Pauli(b1, b1) = 0;
            Pauli(b0, b1) = -rmi;
            Pauli(b1, b0) = rmi;
        }
        if(P == 3){ // Z
            Pauli(b0, b0) = 1;
            Pauli(b1, b1) = -1;
            Pauli(b0, b1) = 0;
            Pauli(b1, b0) = 0;
        }
    }
    return Pauli;
}

void apply_two_qubit_time_evol(arma::cx_mat &A, arma::cx_mat &hij, int n, int i, int j, double t){
    arma::cx_mat exp_rmi_t_hij = expmat(rmi * t * hij);

    // Apply exp_rmi_t_hij to A
    arma::cx_mat A_new(1 << n, 1 << n, arma::fill::zeros);

    for(int b = 0; b < (1 << n); b++){
        int bij = (((b & (1 << i)) >> i) * 2) + ((b & (1 << j)) >> j);
        for(int q = 0; q < (1 << n); q++){
            int qij = (((q & (1 << i)) >> i) * 2) + ((q & (1 << j)) >> j);
            A_new(b, q) = 0.0;

            for(int r = 0; r < 4; r++){
                int br = (b - (b & (1 << i)) - (b & (1 << j))) + ((r >> 1) << i) + ((r & 1) << j);
                for(int s = 0; s < 4; s++){
                    int qs = (q - (q & (1 << i)) - (q & (1 << j))) + ((s >> 1) << i) + ((s & 1) << j);

                    A_new(b, q) += exp_rmi_t_hij(bij, r) * A(br, qs) * conj(exp_rmi_t_hij(qij, s));
                }
            }
        }
    }
    // debug purpose
    // cout << A_new << endl;
    // cout << exp_rmi_t_hij * A * exp_rmi_t_hij.adjoint() << endl;

    A = A_new;

    A_new.reset();
    exp_rmi_t_hij.reset();
}

void remove_a_qubit(arma::cx_mat &A, int n, int i){
    arma::cx_mat A_new(1 << (n-1), 1 << (n-1));

    for(int b = 0; b < (1 << n); b++){
        int bi = (b & (1 << i));
        int bnoti = b ^ (1 << i);
        int bremi = ((b >> (i+1)) << i) + (b & ((1 << i) - 1));

        for(int q = 0; q < (1 << n); q++){
            int qi = (q & (1 << i));
            int qnoti = q ^ (1 << i);
            int qremi = ((q >> (i+1)) << i) + (q & ((1 << i) - 1));

            if(bi == qi)
                A_new(bremi, qremi) = (A(b, q) + A(bnoti, qnoti)) / 2.0;
        }
    }

    A = A_new;
    A_new.reset();
}

double remove_a_qubit_put_an_I(arma::cx_mat &A, int n, int i){
    arma::cx_mat A_new(1 << n, 1 << n);

    for(int b = 0; b < (1 << n); b++){
        int bi = (b & (1 << i));
        int bnoti = b ^ (1 << i);

        for(int q = 0; q < (1 << n); q++){
            int qi = (q & (1 << i));
            int qnoti = q ^ (1 << i);

            if(bi == qi)
                A_new(b, q) = (A(b, q) + A(bnoti, qnoti)) / 2.0;
            else
                A_new(b, q) = 0;
        }
    }

    double diff = arma::norm(A - A_new);

    A = A_new;
    A_new.reset();

    return diff;
}
