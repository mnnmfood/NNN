#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>

#include <Eigen/Dense>

#include "sequential.h"
#include "activations.h"
#include "costs.h"

using Eigen::Matrix;
using Eigen::seqN;
using Eigen::all;
using Eigen::last;
using Eigen::Vector;
using Eigen::Dynamic;


// for weight initialization
NormalSample::NormalSample(float _mean, float _std): nd(_mean, _std){}
float NormalSample::operator()(){return static_cast<float>(nd(gen));}

Sequential::Sequential(std::vector<int>& _arch, ActivationFun<float>* a, CostFun* c,
                        bool use_softmax=false)
    :num_layers{_arch.size()}, arch{_arch}, activationFun{a}, costFun{c}
{
    int n_last{arch.front()};

    for(std::vector<int>::iterator it {arch.begin() + 1}; it != arch.end(); it++){

        NormalSample sampleFun(0.0f, 1.0f / std::sqrt(n_last));

        weights.push_back( Matrix<float, Dynamic, Dynamic>(*it, n_last));
        //weights.back()->setRandom();
        //weights.back() = Matrix<float, Dynamic, Dynamic>::Random(*it, n_last);
        weights.back() = (weights.back()).NullaryExpr(*it, n_last,
            std::ref(sampleFun));

        biases.push_back( Vector<float, Dynamic>(*it));
        //biases.back() = Vector<float, Dynamic>::Random(*it);
        biases.back() = (biases.back()).NullaryExpr(*it,
            std::ref(sampleFun));
        //biases.back()->setRandom();

        nabla_b.push_back( Vector<float, Dynamic>(*it));
        nabla_w.push_back( Matrix<float, Dynamic, Dynamic>(*it, n_last));

        activationFun.push_back(a);
        n_last = *it;
    }
    std::cout << "fun layers" << activationFun.size() << "\n";
    if(use_softmax){
        std::cout << "fun layers" << activationFun.size() << "\n";
            activationFun.back() = new SoftMax<float>();
    }
}

void Sequential::feedFwd(const Matrix<float, Dynamic, Dynamic>& input){
    activations[0] = input;

    w_inputs[0] = (weights[0] * input).colwise() + biases[0];
    activations[1] = activationFun[0]->activation(w_inputs[0]);

    for(size_t i{1}; i < num_layers-1; i++){
        w_inputs[i] = (weights[i] * activations[i]).
                        colwise() + biases[i];      
        activations[i+1] = activationFun[i]->activation(w_inputs[i]);
    }
}

void Sequential::backProp(const Matrix<float, Dynamic, Dynamic>& x, const Matrix<float, Dynamic, Dynamic>& y){

    feedFwd(x);

    //delta.back() = costFun->grad(activations.back(), y).cwiseProduct(
    //        activationFun[num_layers - i + 1]->activation_prime(w_inputs.back()));
    delta.back() = costFun->grad(activations.back(), y,
            activationFun.back()->activation_prime(w_inputs.back()));
    nabla_b.back() = (delta.back()).rowwise().sum();
    nabla_w.back() = delta.back() * (activations[num_layers-2]).transpose();


    for(size_t i{3}; i < num_layers+1; i++){
        delta[num_layers-i] = ((weights[num_layers -i + 1]).transpose() * 
            delta[num_layers - i + 1]).cwiseProduct(activationFun[num_layers - i]->
            activation_prime(w_inputs[num_layers - i]));

        nabla_b[num_layers-i] = (delta[num_layers-i]).rowwise().sum();
        nabla_w[num_layers-i] = delta[num_layers-i] *
            ((activations[num_layers-i]).transpose());
    }
}

void Sequential::initGD(size_t n_samples){
    activations.push_back(
            Matrix<float, Dynamic, Dynamic>(arch[0], n_samples));
    for(size_t i{1}; i < num_layers; i++){
        delta.push_back(
                Matrix<float, Dynamic, Dynamic>(arch[i], n_samples));
        activations.push_back(
                Matrix<float, Dynamic, Dynamic>(arch[i], n_samples));
        w_inputs.push_back(
                Matrix<float, Dynamic, Dynamic>(arch[i], n_samples));
    }
}

void Sequential::GD(Matrix<float, Dynamic, Dynamic>& x,
        Matrix<float, Dynamic, Dynamic>& y, 
        int epochs, float lr, float eta)
{
    size_t n_samples = x.cols();
    initGD(n_samples);

    for(int k{0}; k < epochs; k++){
        backProp(x, y);

        for(size_t i{0}; i < num_layers-1; i++){
            weights[i] = weights[i].eval()*(1 - lr*eta/n_samples) - 
                (lr / n_samples) * nabla_w[i];
            biases[i] -= (lr / n_samples) * nabla_b[i];
        }
    }
}


void Sequential::GD(Matrix<float, Dynamic, Dynamic>& x,
    Matrix<float, Dynamic, Dynamic>& y, 
    int epochs, float lr, float eta,
    Matrix<float, Dynamic, Dynamic>& val_x,
    Matrix<float, Dynamic, Dynamic>& val_y)
{
    size_t n_samples = x.cols();
    initGD(n_samples);

    float cost_t{0};
    for(int k{0}; k < epochs; k++){
        backProp(x, y);

        for(size_t i{0}; i < num_layers-1; i++){
            weights[i] = weights[i].eval()*(1 - lr*eta/n_samples) - 
                (lr / n_samples) * nabla_w[i];
            biases[i] -= (lr / n_samples) * nabla_b[i];
        }

        cost_t = accuracy(val_x, val_y);
        std::cout << "Accuracy " << k << " :" << cost_t  ;
        std::cout << "%" << "\n";
    }
}

void Sequential::SGD(Matrix<float, Dynamic, Dynamic>& x,
        Matrix<float, Dynamic, Dynamic>& y, 
        int epochs, int batch_size, float lr, float eta)
{
    size_t train_size = x.cols();
    initGD(batch_size);

    // Prepare generator of random indices 
    std::vector<int> indices;
    std::vector<int> sub_indices(batch_size);
    for(size_t i{0}; i < train_size; i++){indices.push_back(i);} 

    float cost_t;

    for(int k{0}; k < epochs; k++){

        // Get random subsample
        std::shuffle(indices.begin(), indices.end(), gen);
        std::copy_n(indices.begin(), batch_size, sub_indices.begin());
        backProp(x(all, sub_indices), y(all, sub_indices));

        for(size_t i{0}; i < num_layers-1; i++){
            weights[i] = weights[i].eval()*(1 - lr*eta/batch_size) - 
                (lr / batch_size) * nabla_w[i];
            biases[i] -= (lr / batch_size) * nabla_b[i];

        }

        feedFwd(x(all, sub_indices));
        cost_t = costFun->cost(activations.back(), y(all, sub_indices));
        //cost_t = accuracy(val_x, val_y);
        std::cout << "Cost " << k << " :" << cost_t / batch_size << "\n";
    }
}


void Sequential::SGD(Matrix<float, Dynamic, Dynamic>& x,
        Matrix<float, Dynamic, Dynamic>& y, 
        int epochs, int batch_size, float lr, float eta,
        Matrix<float, Dynamic, Dynamic>& val_x,
        Matrix<float, Dynamic, Dynamic>&val_y)
{

    size_t train_size = x.cols();
    initGD(batch_size);

    // Prepare random indices 
    std::vector<int> indices;
    std::vector<int> sub_indices(batch_size);
    for(size_t i{0}; i < train_size; i++){indices.push_back(i);} 

    float cost_t;
    for(int k{0}; k < epochs; k++){
        auto start = std::chrono::high_resolution_clock::now();
        // Get random subsample
        std::shuffle(indices.begin(), indices.end(), gen);
        for(size_t l{0}; l < train_size-batch_size; l+=batch_size){
            std::copy_n(indices.begin()+l, batch_size, sub_indices.begin());
            backProp(x(all, sub_indices), y(all, sub_indices));
            for(size_t i{0}; i < num_layers-1; i++){
                weights[i] = weights[i].eval()*(1 - lr*eta/batch_size) - 
                    (lr / batch_size) * nabla_w[i];
                biases[i] -= (lr / batch_size) * nabla_b[i];
            }

            //cost_t = costFun->cost(activations.back(), y(all, sub_indices));
        }

        cost_t = accuracy(val_x, val_y);
        std::cout << "Epoch " << k << " :" << cost_t*100; 
        std::cout << " %" << "\n";
        auto finish = std::chrono::high_resolution_clock::now();
        std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(finish-start).count() << "ms\n";
    }
}

float Sequential::accuracy(const Matrix<float, Dynamic, Dynamic>& x,
                const Matrix<float, Dynamic, Dynamic>& y)
{
    
    Eigen::Index test_size{x.cols()};
    feedFwd(x);

    int y_pred;
    int sum{0};
    for(Eigen::Index i{0}; i < test_size; i++){
        //y.col(i).maxCoeff(&y_test);
        (activations.back()).col(i).maxCoeff(&y_pred);
        //std::cout << y_pred << "\n";
        //std::cout << activations.back().col(i) << "\n\n";
        sum += (static_cast<int>(y(0, i)) == y_pred);
    }
    return static_cast<float>(sum) / static_cast<float>(test_size);
}

std::ostream& operator<<(std::ostream& out, const Sequential& model) 
{
        out << "Layer" << '\t' << "Nodes" << '\t' << "Weights" << '\n';
        size_t l = model.weights.size();
        out << 0 << '\t' << model.weights[0].cols() << '\t' << 0 << '\n';
        for(size_t i{0}; i < l; i++){
            out << i+1 << '\t' << model.biases[i].size() << 
            '\t' << model.weights[i].size() << '\n';
        }
        return out;
}