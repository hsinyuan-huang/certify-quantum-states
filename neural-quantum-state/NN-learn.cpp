#include <cstdio>
#include <iostream>
#include <fstream>
#include <bitset>
#include <vector>
#include <random>
#include <complex>
#include <cmath>
#include <map>
#include <algorithm>

#include "neural_network.h"
const double EPS = 1e-10;

using namespace std;

// Learning neural binary phase states
int n, N, type;
int seed_cnt = 10;
int binary_phase;

class Model{
private:
    vector<LinearLayer> ff;
    vector<Sigmoid> act;
    vector<double> output_vals;
public:
    Model(int n, int h, int target);
    vector<double> feedForward(vector<double> input, bool eval = 0);
    void train(vector<double> input, int xy_basis, double outcome);
    double predict_phase_diff(vector<double> input);
    double log_loss(vector<double> input, int xy_basis, double outcome);
    double shadow_fidelity_est(vector<double> input, int xy_basis, double outcome);
};
Model::Model(int n, int h, int target){
    ff.push_back(LinearLayer(n, h, 0.05, 0));
    act.push_back(Sigmoid());
    ff.push_back(LinearLayer(h, 2, 0.05, 0));
    act.push_back(Sigmoid());
}
vector<double> Model::feedForward(vector<double> input, bool eval){
    for(int i = 0; i < ff.size(); i++){
        input = ff[i].feedForward(input, eval);
        input = act[i].feedForward(input, eval);
    }
    output_vals = input;
    return output_vals;
}
void Model::train(vector<double> input, int xy_basis, double outcome){
    double prob = feedForward(input, 1)[xy_basis];
    double gd = (-outcome / (prob + EPS) + (1.0 - outcome) / (1.0 - prob + EPS));

    for(int i = 0; i < ff.size(); i++)
        ff[i].initGrad();

    vector<double> grad(2, 0);
    feedForward(input, 0);
    grad[xy_basis] = gd;

    for(int i = (int)ff.size() - 1; i >= 0; i--){
        grad = act[i].backPropagate(grad);
        grad = ff[i].backPropagate(grad, 0);
    }

    for(int i = 0; i < ff.size(); i++)
        ff[i].stepGrad();
}
double Model::log_loss(vector<double> input, int xy_basis, double outcome){
    double prob = feedForward(input, 1)[xy_basis];
    return -outcome * log(prob + EPS) - (1.0 - outcome) * log(1.0 - prob + EPS);
}
double Model::predict_phase_diff(vector<double> input){
    vector<double> prob_vec = feedForward(input, 1);

    double xcoord = (2 * prob_vec[0] - 1);
    double ycoord = (2 * prob_vec[1] - 1);
    if(binary_phase) ycoord = 0;

    if(abs(xcoord) > 1.5 * abs(ycoord)){
        double ylim = sqrt(1.0 - xcoord * xcoord);
        ycoord = max(min(ycoord, ylim), -ylim);
    }
    else if(abs(ycoord) > 1.5 * abs(xcoord)){
        double xlim = sqrt(1.0 - ycoord * ycoord);
        xcoord = max(min(xcoord, xlim), -xlim);
    }

    complex<double> cplx_num = xcoord + rmi * ycoord;
    return imag(log(cplx_num));
}
double Model::shadow_fidelity_est(vector<double> input, int xy_basis, double outcome){
    double phase_diff = predict_phase_diff(input);
    if(xy_basis == 0 && outcome == 1)
        phase_diff -= 0;
    if(xy_basis == 0 && outcome == 0)
        phase_diff -= M_PI;
    if(xy_basis == 1 && outcome == 1)
        phase_diff -= M_PI / 2.0;
    if(xy_basis == 1 && outcome == 0)
        phase_diff += M_PI / 2.0;
    return 2.0 / 3.0 * (3 * pow(abs(0.5 + 0.5 * exp(rmi * phase_diff)), 2) - 1) + 1.0 / 3.0 * 0.5;
}

map<vector<int>, double> random_binary_phase;
double pseudo_random_state_phase(vector<double> input, int seed){
    vector<int> input_int;
    for(auto x : input) input_int.push_back((int)(x));
    input_int.push_back(seed);
    if(random_binary_phase.count(input_int) == 0)
        random_binary_phase[input_int] = (normal(gen) > 0 ? 1 : 0);
    return random_binary_phase[input_int];
}

vector<int> index_i;
vector<int> index_j;
vector<double> random_ij_phase;

double true_phase(vector<double> b){
    if(type == 0){ // type 0: pseudorandom binary phase state
        vector<int> list_phase;
        for(int seed = 0; seed < seed_cnt; seed++)
            list_phase.push_back(pseudo_random_state_phase(b, seed));
        return list_phase[seed_cnt / 2] * M_PI;
    }
    else{ // type 1: low degree phase state
        double phase = 0;
        for(int c = 0; c < (int)index_i.size(); c++){
            if(b[index_i[c]] > 0.5 && b[index_j[c]] > 0.5)
                phase += random_ij_phase[c];
        }
        return phase;
    }
}

vector<double> random_init_state;
double predict_phase(Model &predictor, vector<double> &bitstr){
    if(random_init_state.empty()){
        for(int i = 0; i < n; i++)
            random_init_state.push_back((int)(unif(gen) * 2));
    }

    vector<double> bitstr_cur = random_init_state;

    double phase = 0;
    for(int i = 0; i < n; i++){
        if(bitstr[i] != random_init_state[i]){
            vector<double> b0 = bitstr_cur;
            bitstr_cur[i] = bitstr[i];
            vector<double> b1 = bitstr_cur;

            if(b0[i] == 1) swap(b0, b1);

            vector<double> feat_vec = b0;
            for(int j = 0; j < n; j++)
                feat_vec.push_back( j == i ? 1 : 0 );
            for(int seed = 0; seed < seed_cnt; seed++){
                feat_vec.push_back(pseudo_random_state_phase(b0, seed));
                feat_vec.push_back(pseudo_random_state_phase(b1, seed));
            }

            double phase_diff = predictor.predict_phase_diff(feat_vec);
            if(bitstr[i] == 0)
                phase += phase_diff;
            else
                phase -= phase_diff;
        }
    }

    return phase;
}

vector<vector<double> > b_list;
vector<int> i_list;
vector<int> xy_list;
vector<double> o_list;

vector<vector<double> > b_test_list;
vector<int> random_i_test_list;

double fidelity(Model phase_predictor){
    if(b_test_list.empty()){
        for(int r = 0; r < 10000; r++){
            vector<double> b;
            for(int i = 0; i < n; i++)
                b.push_back(int(unif(gen) * 2));
            b_test_list.push_back(b);

            random_i_test_list.push_back(int(unif(gen) * n));
        }
    }

    complex<double> fid = 0.0;
    for(int r = 0; r < 10000; r++){
        vector<double> b = b_test_list[r];
        fid += exp(rmi * (predict_phase(phase_predictor, b) - true_phase(b))) * (1.0 / 10000);
    }
    return abs(fid) * abs(fid);
}

double shadow_fidelity(Model phase_predictor){
    if(b_test_list.empty()){
        for(int r = 0; r < 10000; r++){
            vector<double> b;
            for(int i = 0; i < n; i++)
                b.push_back(int(unif(gen) * 2));
            b_test_list.push_back(b);

            random_i_test_list.push_back(int(unif(gen) * n));
        }
    }

    double Sha_Fid = 0.0;

    for(int r = 0; r < 10000; r++){
        vector<double> b = b_test_list[r];
        int random_i = random_i_test_list[r];
        vector<double> b0 = b; b0[random_i] = 0;
        vector<double> b1 = b; b1[random_i] = 1;

        vector<double> feat_vec = b0;
        for(int i = 0; i < n; i++)
            feat_vec.push_back( i == random_i ? 1 : 0 );
        for(int seed = 0; seed < seed_cnt; seed++){
            feat_vec.push_back(pseudo_random_state_phase(b0, seed));
            feat_vec.push_back(pseudo_random_state_phase(b1, seed));
        }

        double true_phase_diff = true_phase(b0) - true_phase(b1);
        Sha_Fid += pow(abs(0.5 + 0.5 * exp(rmi * (phase_predictor.predict_phase_diff(feat_vec) - true_phase_diff))), 2);
    }
    Sha_Fid /= 10000;

    return 1.0 * (1.0 - pow(0.5, n)) * 2 * (Sha_Fid - 0.5) + pow(0.5, n);
}

double train_loss(Model phase_predictor){
    double loss = 0.0;

    for(int r = 0; r < N / 2; r++){
        vector<double> b0 = b_list[r]; b0[i_list[r]] = 0;
        vector<double> b1 = b_list[r]; b1[i_list[r]] = 1;

        vector<double> feat_vec = b0;
        for(int i = 0; i < n; i++)
            feat_vec.push_back( i == i_list[r] ? 1 : 0 );
        for(int seed = 0; seed < seed_cnt; seed++){
            feat_vec.push_back(pseudo_random_state_phase(b0, seed));
            feat_vec.push_back(pseudo_random_state_phase(b1, seed));
        }

        loss += phase_predictor.log_loss(feat_vec, xy_list[r], o_list[r]);
    }

    return loss / (N / 2);
}

double val_loss(Model phase_predictor){
    double loss = 0.0;

    for(int r = N / 2; r < N; r++){
        vector<double> b0 = b_list[r]; b0[i_list[r]] = 0;
        vector<double> b1 = b_list[r]; b1[i_list[r]] = 1;

        vector<double> feat_vec = b0;
        for(int i = 0; i < n; i++)
            feat_vec.push_back( i == i_list[r] ? 1 : 0 );
        for(int seed = 0; seed < seed_cnt; seed++){
            feat_vec.push_back(pseudo_random_state_phase(b0, seed));
            feat_vec.push_back(pseudo_random_state_phase(b1, seed));
        }

        loss += phase_predictor.log_loss(feat_vec, xy_list[r], o_list[r]);
    }

    return loss / (N - (N / 2));
}

double shadow_fidelity_tra(Model phase_predictor){
    double Sha_Fid = 0.0;

    for(int r = 0; r < N / 2; r++){
        vector<double> b0 = b_list[r]; b0[i_list[r]] = 0;
        vector<double> b1 = b_list[r]; b1[i_list[r]] = 1;

        vector<double> feat_vec = b0;
        for(int i = 0; i < n; i++)
            feat_vec.push_back( i == i_list[r] ? 1 : 0 );
        for(int seed = 0; seed < seed_cnt; seed++){
            feat_vec.push_back(pseudo_random_state_phase(b0, seed));
            feat_vec.push_back(pseudo_random_state_phase(b1, seed));
        }

        Sha_Fid += phase_predictor.shadow_fidelity_est(feat_vec, xy_list[r], o_list[r]);
    }
    Sha_Fid /= (N / 2);

    return 1.0 * (1.0 - pow(0.5, n)) * 2 * (Sha_Fid - 0.5) + pow(0.5, n);
}

double shadow_fidelity_val(Model phase_predictor){
    double Sha_Fid = 0.0;

    for(int r = N / 2; r < N; r++){
        vector<double> b0 = b_list[r]; b0[i_list[r]] = 0;
        vector<double> b1 = b_list[r]; b1[i_list[r]] = 1;

        vector<double> feat_vec = b0;
        for(int i = 0; i < n; i++)
            feat_vec.push_back( i == i_list[r] ? 1 : 0 );
        for(int seed = 0; seed < seed_cnt; seed++){
            feat_vec.push_back(pseudo_random_state_phase(b0, seed));
            feat_vec.push_back(pseudo_random_state_phase(b1, seed));
        }

        Sha_Fid += phase_predictor.shadow_fidelity_est(feat_vec, xy_list[r], o_list[r]);
    }
    Sha_Fid /= (N - (N / 2));

    return 1.0 * (1.0 - pow(0.5, n)) * 2 * (Sha_Fid - 0.5) + pow(0.5, n);
}

vector<vector<double> > b1_list;
vector<vector<double> > b2_list;

double purity(Model phase_predictor, int less_than_this, int truth = 0){
    if(b1_list.empty()){
        for(int r = 0; r < 30000; r++){
            vector<double> b1;
            for(int i = 0; i < n; i++)
                b1.push_back(int(unif(gen) * 2));
            vector<double> b2;
            for(int i = 0; i < n; i++)
                b2.push_back(int(unif(gen) * 2));
            b1_list.push_back(b1);
            b2_list.push_back(b2);
        }
    }

    complex<double> purity = 0.0;

    for(int r = 0; r < 30000; r++){
        vector<double> b1 = b1_list[r];
        vector<double> b2 = b2_list[r];

        vector<double> b1_alt = b1, b2_alt = b2;
        for(int i = 0; i < less_than_this; i++)
            swap(b1_alt[i], b2_alt[i]);

        double phase1, phase2, phase1_alt, phase2_alt;

        if(truth == 0){
            phase1 = predict_phase(phase_predictor, b1);
            phase2 = predict_phase(phase_predictor, b2);
            phase1_alt = predict_phase(phase_predictor, b1_alt);
            phase2_alt = predict_phase(phase_predictor, b2_alt);
        }
        else{
            phase1 = true_phase(b1);
            phase2 = true_phase(b2);
            phase1_alt = true_phase(b1_alt);
            phase2_alt = true_phase(b2_alt);
        }

        purity += exp(rmi * (phase1_alt + phase2_alt - phase1 - phase2)) / 30000.0;
    }

    return purity.real();
}


//
// Run the commandlines:
//    ./NN-learn 120 50000 0 10 1
//    ./NN-learn 40 1000000 1 0 0
//
int main(int argc, char *argv[]){
    if (argc != 6) {
        cout << "./NN-learn n N type seed_cnt binary_phase" << endl;
        return 1;
    }

    n = atoi(argv[1]);
    N = atoi(argv[2]);
    type = atoi(argv[3]); // type 0: pseudorandom, type 1: correlated state
    seed_cnt = atoi(argv[4]);
    binary_phase = atoi(argv[5]); // 1: the model only predicts binary phase (+1, -1) else: 0
    if(type == 1){
        for(int s = 0; s < n-1; s++){
            index_i.push_back(s);
            index_j.push_back((s+10) % n);
            random_ij_phase.push_back(0.5 * M_PI);

            if(s % 10 == 9) s += 10;
        }
    }

    Model phase_predictor(2*(n+seed_cnt), 4 * n, 0);

    for(int r = 0; r < N; r++){
        vector<double> one_b;
        for(int i = 0; i < n; i++)
            one_b.push_back(int(unif(gen) * 2));

        b_list.push_back(one_b);
        i_list.push_back(int(unif(gen) * n));
        xy_list.push_back(int(unif(gen) * 2)); // 0: X basis, 1: Y basis

        vector<double> b0 = one_b; b0[i_list.back()] = 0;
        vector<double> b1 = one_b; b1[i_list.back()] = 1;
        double phase0 = true_phase(b0);
        double phase1 = true_phase(b1);

        if(xy_list.back() == 0){
            if((1+cos(phase0 - phase1)) / 2.0 > unif(gen))
                o_list.push_back(1); // |+>
            else
                o_list.push_back(0); // |->
        }
        else{
            if((1+sin(phase0 - phase1)) / 2.0 > unif(gen))
                o_list.push_back(1); // |y+>
            else
                o_list.push_back(0); // |y->
        }
    }


    Model best_predictor = phase_predictor;
    double best_val_loss = val_loss(phase_predictor);

    // cout << "Subsystem Purity (Truth): " << endl;
    // for(int k = 0; k <= n; k++)
    //     cout << purity(phase_predictor, k, 1) << " " << flush;
    // cout << endl;
    //
    // cout << "Subsystem Purity (Random init NN): " << endl;
    // for(int k = 0; k <= n; k++)
    //     cout << purity(phase_predictor, k, 0) << " " << flush;
    // cout << endl;

    int cnt = 0, tick = 0, steps = 0;
    cout << "0 Best" << endl;
    cout << "Tlogloss: " << train_loss(phase_predictor) << endl;
    cout << "Vlogloss: " << best_val_loss << endl;
    cout << "TShadowF: " << shadow_fidelity_tra(phase_predictor) << endl;
    cout << "VShadowF: " << shadow_fidelity_val(phase_predictor) << endl;
    cout << "Shadow_F: " << shadow_fidelity(phase_predictor) << endl;
    cout << "Fidelity: " << fidelity(phase_predictor) << endl;
    cnt ++;

    int intv_cnt, max_intv_cnt, intv_tick, num_epoch;
    if(type == 0){
        intv_cnt = N / 20;
        max_intv_cnt = N / 20;
        intv_tick = 8;
        num_epoch = 10;
    }
    if(type == 1){
        intv_cnt = 1;
        max_intv_cnt = N / 20;
        intv_tick = 8;
        num_epoch = 9;
    }

    for(int epoch = 0; epoch < num_epoch; epoch++){
        for(int r = 0; r < N / 2; r++){
            vector<double> b0 = b_list[r]; b0[i_list[r]] = 0;
            vector<double> b1 = b_list[r]; b1[i_list[r]] = 1;

            vector<double> feat_vec = b0;
            for(int i = 0; i < n; i++)
                feat_vec.push_back( i == i_list[r] ? 1 : 0 );
            for(int seed = 0; seed < seed_cnt; seed++){
                feat_vec.push_back(pseudo_random_state_phase(b0, seed));
                feat_vec.push_back(pseudo_random_state_phase(b1, seed));
            }

            phase_predictor.train(feat_vec, xy_list[r], o_list[r]);
            steps ++;

            if(cnt == intv_cnt){
                double valoss = val_loss(phase_predictor);
                if(best_val_loss > valoss){
                    best_val_loss = valoss;
                    best_predictor = phase_predictor;
                    cout << steps << " Best" << endl;
                }
                else
                    cout << steps << " NotBest" << endl;

                cout << "Tlogloss: " << train_loss(phase_predictor) << endl;
                cout << "Vlogloss: " << valoss << endl;
                cout << "TShadowF: " << shadow_fidelity_tra(phase_predictor) << endl;
                cout << "VShadowF: " << shadow_fidelity_val(phase_predictor) << endl;
                cout << "Shadow_F: " << shadow_fidelity(phase_predictor) << endl;

                tick ++;
                if(tick == intv_tick){
                    cout << "Fidelity: " << fidelity(phase_predictor) << endl;
                    tick = 0;
                }
                else{
                    cout << "Fidelity: " << -999999 << endl;
                }

                intv_cnt = min(intv_cnt * 2, max_intv_cnt);
                cnt = 0;
            }
            else cnt ++;
        }
    }

    // cout << "Subsystem Purity (Trained NN): " << endl;
    // for(int k = 0; k <= n; k++)
    //     cout << purity(best_predictor, k, 0) << " " << flush;
    // cout << endl;
    cout << "Fidelity (Trained NN): " << fidelity(best_predictor) << endl;

    return 0;
}
