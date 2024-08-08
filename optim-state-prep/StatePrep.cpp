#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <math.h>
#include <complex>

using namespace std;

int n; // system size
const complex<double> rmi(0.0, 1.0); // complex i

//
// Target state is the output of a 1D IQP circuit:
// HHH...HHH (\Pi_i T^{random_T_pattern[i]}) (\Pi_i CZ_{i, i+1}) HHH...HHH |000...000>
//
// We consider random_T_pattern[i] to be {-1, 0, 1}
//

vector<int> random_T_pattern;

double estimate_fidelity(vector<int> seq_action)
{
	complex<double> fidel = 0.0;
	for (int rep = 0; rep < 10000; rep++)
	{
		vector<int> bitstring(n, 0);
		for (int i = 0; i < n; i++)
		{
			int mi = rand() % 2;
			bitstring[i] = mi;
		}

		int how_many_T = 0;
		for (auto act : seq_action)
		{
			int pos = act % n;
			int T_or_not = act / n;

			if (T_or_not == 1)
			{
				if (bitstring[pos] == 1)
					how_many_T += 1;
			}
			else if (T_or_not == 2)
			{
				if (bitstring[pos] == 1)
					how_many_T = (how_many_T - 1 + 8) % 8;
			}
			else{
				if (bitstring[pos] == 1 && bitstring[(pos + 1) % n] == 1)
					how_many_T += 4;
			}
		}
		int how_many_T_true = 0;
		for (int i = 0; i < n; i++)
		{
			if (bitstring[i] == 1 && bitstring[(i + 1) % n] == 1)
				how_many_T_true += 4;
			if (bitstring[i] == 1)
				how_many_T_true = (how_many_T_true + random_T_pattern[i] + 8) % 8;
		}

		int phase_diff = (how_many_T - how_many_T_true + 8) % 8;
		fidel += exp(rmi * (phase_diff / 8.0) * 2.0 * M_PI) / 10000.0;
	}

	return abs(fidel) * abs(fidel);
}

//
// Shadow overlap is measured in all X bases except for qubit random_x
//
double estimate_one_shadow_overlap(vector<int> seq_action){
	int random_x = rand() % n;
	vector<int> bitstring(n, 0);

	for(int i = 0; i < n; i++){
		if(i != random_x){
			int mi = rand() % 2;
			bitstring[i] = mi;
		}
	}

	int how_many_T = 0;
	for(auto act : seq_action){
		int pos = act % n;
		int T_or_not = act / n;

		if (T_or_not == 1)
		{
			if (pos == random_x)
				 how_many_T += 1;
		}
		else if (T_or_not == 2)
		{
			if (pos == random_x)
				how_many_T = (how_many_T - 1 + 8) % 8;
		}
		else{
			if (pos == (random_x - 1 + n) % n && bitstring[(random_x - 1 + n) % n] == 1)
				how_many_T += 4;
			if (pos == random_x && bitstring[(random_x + 1) % n] == 1)
				how_many_T += 4;
		}
	}

	//
	// A simplified 1D cycle MPS contraction
	//
	// The contracted target 1-qubit state on the random_X qubit is
	//    1/sqrt(2) |0> + 1/sqrt(2) exp(i 2 pi (how_many_T_true/8)) |1>
	//
	// This is obtained by the structure of the target state
	//
	int how_many_T_true = 0;
	if (bitstring[(random_x - 1 + n) % n] == 1)
		how_many_T_true += 4;
	if (bitstring[(random_x + 1) % n] == 1)
		how_many_T_true += 4;
	how_many_T_true = (how_many_T_true + random_T_pattern[random_x] + 8) % 8;

	int phase_diff = (how_many_T - how_many_T_true + 8) % 8;
	return abs(0.5 + 0.5 * exp(rmi * (phase_diff / 8.0) * 2.0 * M_PI))
	* abs(0.5 + 0.5 * exp(rmi * (phase_diff / 8.0) * 2.0 * M_PI));
}

//
// Integer for each action:
// 	 0 * n, ..., 1 * n - 1: HH CZ HH
// 	 1 * n, ..., 2 * n - 1: H T H
// 	 2 * n, ..., 3 * n - 1: H T^-1 H
//

int main(int argc, char **argv){
	if(argc < 4){
		cout << "Usage: ./StatePrep <n> <seed> <type>" << endl;
		return 0;
	}
	// Type = 0: Train with fidelity
	// Type = 1: Train with shadow overlap
	// Type = 2: Plot state construction process

	n = atoi(argv[1]);
	int seed = atoi(argv[2]);
    srand(seed);
	int type = atoi(argv[3]);

	for(int i = 0; i < n; i++){
		int rand3 = rand() % 3;
		random_T_pattern.push_back(rand3-1); // -1, 0, 1
	}

	vector<int> seq_action;

	if(type < 2){
		double cur_score = -9999;
		double cur_fidel = 0.0;
		double cur_shadow_o = 0.0;

		for (int r = 0; r < 3 * n; r++){
			int best_act = 0;
			double best_one_step = -9999;
			double best_one_step_fidel = 0.0;
			double best_one_step_shadow_o = 0.0;

			for(int act = 0; act < 3 * n; act++){
				seq_action.push_back(act);

				double shadow_o = 0.0;
				for (int rep = 0; rep < 10000; rep++)
					shadow_o += estimate_one_shadow_overlap(seq_action) / 10000;
				double fidel = estimate_fidelity(seq_action);
				double score = (type == 0 ? fidel : shadow_o);

				if (best_one_step < score){
					best_one_step = score;
					best_act = act;
					best_one_step_fidel = fidel;
					best_one_step_shadow_o = shadow_o;
				}
				seq_action.pop_back();
			}

			if (cur_score < best_one_step || cur_score < 0.99)
			{
				seq_action.push_back(best_act);
				cur_score = best_one_step;
				cur_fidel = best_one_step_fidel;
				cur_shadow_o = best_one_step_shadow_o;
			}

			cout << cur_fidel << " " << 2 * (cur_shadow_o - 0.5) << endl;
		}
	}
	else{
		for (int i = 0; i < n; i++)
		{
			seq_action.push_back(i);
			if(random_T_pattern[i] != 0)
				seq_action.push_back(((random_T_pattern[i] + 3) % 3) * n + i);
			double shadow_o = 0.0;
			for (int rep = 0; rep < 10000; rep++)
				shadow_o += estimate_one_shadow_overlap(seq_action) / 10000;
			double fidel = estimate_fidelity(seq_action);
			cout << fidel << " " << 2 * (shadow_o - 0.5) << endl;
		}
	}

	return 0;
}