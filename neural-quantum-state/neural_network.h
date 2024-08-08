#include <iostream>
#include <vector>
#include <assert.h>
#include <math.h>
#include "circuit_simulation.h"
using namespace std;

class LinearLayer{
private:
    vector<vector<double>> weights;
    vector<vector<double>> stored_grad;
    vector<double> output_vals;
    vector<double> input_vals;
    int output_dim;
    int input_dim;
    double eta;
public:
    LinearLayer(){}
    LinearLayer(int input_size, int output_size, double lr, int large_var);
    void initGrad();
    void stepGrad();
    vector<double> feedForward(const vector<double> &input, bool eval=0);
    vector<double> backPropagate(const vector<double> &grad, bool eval=0);
};
LinearLayer::LinearLayer(int input_size, int output_size, double lr, int large_var){
    assert(input_size > 0);
    assert(output_size > 0);
    output_dim = output_size;
    input_dim = input_size;
    eta = lr;

    //generate random weights
    for(int out = 0; out < output_size; out++){
        weights.push_back(vector<double>());
        for(int input = 0; input <= input_size; input++){
        //we create an extra weight (one more than input_size) for our bias
            weights.back().push_back(normal(gen) * sqrt(1.0 / (input_dim + output_dim)));
        }
    }
}
void LinearLayer::initGrad(){
    stored_grad.clear();
    for(int out = 0; out < output_dim; out++){
        vector<double> one_grad;
        for(int input = 0; input <= input_dim; input++)
            one_grad.push_back(0);
        stored_grad.push_back(one_grad);
    }
}
void LinearLayer::stepGrad(){
    for(int out = 0; out < output_dim; out++){
        for(int input = 0; input <= input_dim; input++){
            weights[out][input] -= eta * stored_grad[out][input];
        }
    }
}
vector<double> LinearLayer::feedForward(const vector<double> &input, bool eval){
    assert(input.size() == input_dim);
    vector<double> output;

    //perform matrix multiplication
    for(int out = 0; out < output_dim; out++){
        double sum = 0.0;
        for(int w = 0; w < input_dim; w++){
            sum += weights[out][w] * input[w];
        }
        sum += weights[out][input_dim];
        output.push_back(sum);
    }

    if(eval == 0) input_vals = input; //store the input vector
    if(eval == 0) output_vals = output; //store the output vector

    return output;
}
vector<double> LinearLayer::backPropagate(const vector<double> &grad, bool eval){
    assert(grad.size() == output_dim);
    vector<double> prev_layer_grad;

    //calculate partial derivatives with respect to input values
    for(int input = 0; input < input_dim; input++){
        double g = 0.0;
        for(int out = 0; out < output_dim; out++){
            g += (grad[out] * weights[out][input]);
        }
        prev_layer_grad.push_back(g);
    }

    if(eval == 0){
        for(int out = 0; out < output_dim; out++){
            for(int input = 0; input < input_dim; input++){
                stored_grad[out][input] += grad[out] * input_vals[input];
            }
            stored_grad[out][input_dim] += grad[out];
        }
    }

    //return computed partial derivatives to be passed to preceding layer
    return prev_layer_grad;
}

class Sigmoid{
private:
    vector<double> output_vals;
public:
    Sigmoid(){}
    vector<double> feedForward(const vector<double> &input, bool eval = 0);
    vector<double> backPropagate(vector<double> grad);
};
vector<double> Sigmoid::feedForward(const vector<double> &input, bool eval){
    vector<double> output;
    for(int in = 0; in < input.size(); in++)
        output.push_back(1.0 / (1.0 + exp(-input[in])));

    if(eval == 0) output_vals = output;
    return output;
}
vector<double> Sigmoid::backPropagate(vector<double> grad){
    assert(grad.size() == output_vals.size());
    for(int out = 0; out < output_vals.size(); out++)
        grad[out] *= output_vals[out] * (1.0 - output_vals[out]);
    return grad;
}

class ReLU{
private:
    vector<double> output_vals;
public:
    ReLU(){}
    vector<double> feedForward(const vector<double> &input, bool eval = 0);
    vector<double> backPropagate(vector<double> grad);
};
vector<double> ReLU::feedForward(const vector<double> &input, bool eval){
    vector<double> output;

    for(int in = 0; in < input.size(); in++)
        output.push_back(max(input[in], 0.0));

    if(eval == 0) output_vals = output;
    return output;
}
vector<double> ReLU::backPropagate(vector<double> grad){
    assert(grad.size() == output_vals.size());
    for(int out = 0; out < output_vals.size(); out++){
        if(output_vals[out] <= 1e-8)
            grad[out] = 0;
    }
    return grad;
}
