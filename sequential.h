#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

#include <Eigen/Dense>

#include "activations.h"

using Eigen::Matrix;
using Eigen::seqN;
using Eigen::all;
using Eigen::last;
using Eigen::Vector;
using Eigen::Dynamic;

template<typename Scalar>
class Sequential
{
public:
    int num_layers{};
    std::vector<int> arch;

    std::vector<Matrix<Scalar, Dynamic, Dynamic>*> weights;
    std::vector<Vector<Scalar, Dynamic>*> biases;

    std::vector<Matrix<Scalar, Dynamic, Dynamic>*> activations;
    std::vector<Matrix<Scalar, Dynamic, Dynamic>*> w_inputs;

    std::vector<Vector<Scalar, Dynamic>*> nabla_b;
    std::vector<Matrix<Scalar, Dynamic, Dynamic>*> nabla_w;

    std::vector<Matrix<Scalar, Dynamic, Dynamic>*> delta;

    ActivationFun<Scalar>* activationFun;

public:
    Sequential() = delete;
    Sequential(std::vector<int>& arch_in, ActivationFun<Scalar>* a)
    {
        arch = arch_in;
        activationFun = a;
        num_layers = arch.size();
        int n_last{arch.front()};

        for(std::vector<int>::iterator it {arch.begin() + 1}; it != arch.end(); it++){

            weights.push_back(new Matrix<Scalar, Dynamic, Dynamic>(*it, n_last));
            weights.back()->setRandom();

            biases.push_back(new Vector<Scalar, Dynamic>(*it));
            biases.back()->setRandom();

            nabla_b.push_back(new Vector<Scalar, Dynamic>(*it));
            nabla_w.push_back(new Matrix<Scalar, Dynamic, Dynamic>(*it, n_last));

            n_last = *it;
        }
    }

    void feedFwd(const Matrix<Scalar, Dynamic, Dynamic>& input){
        *activations[0] = input;

        *w_inputs[0] = (*weights[0] * input).colwise() + *biases[0];
        *activations[1] = activationFun->activation(*w_inputs[0]);

        for(size_t i{1}; i < num_layers-1; i++){
            *w_inputs[i] = (*weights[i] * *activations[i]).
                            colwise() + *biases[i];      
            *activations[i+1] = activationFun->activation(*w_inputs[i]);
        }
    }

    void backProp(const Matrix<Scalar, Dynamic, Dynamic>& x, 
                 const Matrix<Scalar, Dynamic, Dynamic>& y){
        feedFwd(x);


        *delta.back() = cost_grad(y).cwiseProduct(
                activationFun->activation_prime(*w_inputs.back()));
        *nabla_b.back() = (*delta.back()).rowwise().sum();
        *nabla_w.back() = *delta.back() * (*activations[num_layers-2]).transpose();



        for(int i{3}; i < num_layers+1; i++){
            *delta[num_layers-i] = ((*weights[num_layers -i + 1]).transpose() * 
                *delta[num_layers - i + 1]).cwiseProduct(activationFun->
                activation_prime(*w_inputs[num_layers - i]));

            *nabla_b[num_layers-i] = (*delta[num_layers-i]).rowwise().sum();
            *nabla_w[num_layers-i] = *delta[num_layers-i] *
                ((*activations[num_layers-i]).transpose());
        }
    }

    void initGD(size_t n_samples){
        activations.push_back(
            new Matrix<Scalar, Dynamic, Dynamic>(arch[0], n_samples));
        for(int i{1}; i < num_layers; i++){
            delta.push_back(
                new Matrix<Scalar, Dynamic, Dynamic>(arch[i], n_samples));
            activations.push_back(
                new Matrix<Scalar, Dynamic, Dynamic>(arch[i], n_samples));
            w_inputs.push_back(
                new Matrix<Scalar, Dynamic, Dynamic>(arch[i], n_samples));
        }
    }

    void GD(Matrix<Scalar, Dynamic, Dynamic>& x,
            Matrix<Scalar, Dynamic, Dynamic>& y, 
            int epochs, Scalar lr){
        Scalar n_samples = static_cast<Scalar>(x.cols());
        initGD(n_samples);

        double cost_t{0};
        for(int k{0}; k < epochs; k++){
            backProp(x, y);

            for(int i{0}; i < num_layers-1; i++){
                *weights[i] -= (lr / n_samples) * *nabla_w[i];
                *biases[i] -= (lr / n_samples) * *nabla_b[i];
            }

            feedFwd(x);
            cost_t = cost(y);
            std::cout << "Cost " << k << " :" << cost_t / n_samples << "\n";
        }
    }

    void SGD(Matrix<Scalar, Dynamic, Dynamic>& x,
             Matrix<Scalar, Dynamic, Dynamic>& y, 
            int epochs, int batch_size, Scalar lr){

        size_t train_size = x.cols();
        initGD(batch_size);

        // Prepare generator of random indices 
        std::random_device rd;
        std::mt19937 g(rd());
        std::vector<int> indices;
        std::vector<int> sub_indices(batch_size);
        for(int i{0}; i < train_size; i++){indices.push_back(i);} 

        double cost_t;

        for(int k{0}; k < epochs; k++){

            // Get random subsample
            std::shuffle(indices.begin(), indices.end(), g);
            std::copy_n(indices.begin(), batch_size, sub_indices.begin());
            backProp(x(all, sub_indices), y(all, sub_indices));

            for(int i{0}; i < num_layers-1; i++){
                *weights[i] -= (lr / batch_size) * *nabla_w[i];
                *biases[i] -= (lr / batch_size) * *nabla_b[i];
            }
            
            feedFwd(x(all, sub_indices));
            cost_t = cost(y(all, sub_indices));
            std::cout << "Cost " << k << " :" << cost_t / batch_size << "\n";
        }
    }

    Matrix<Scalar, Dynamic, Dynamic> cost_grad(const Matrix<Scalar, Dynamic, Dynamic>& y){
        return *activations.back() - y;
    } 

    Scalar cost(const Matrix<Scalar, Dynamic, Dynamic>& y){
        return (*activations.back() - y).colwise().squaredNorm().sum();
    }

    Scalar accuracy(const Matrix<Scalar, Dynamic, Dynamic>& x,
                    const Matrix<Scalar, Dynamic, Dynamic>& y){
        
        Eigen::Index test_size{x.cols()};
        feedFwd(x);

        int y_test, y_pred;
        int sum{0};
        for(Eigen::Index i{0}; i < test_size; i++){
            y.col(i).maxCoeff(&y_test);
            (*activations.back()).col(i).maxCoeff(&y_pred);
            std::cout << "Test: " << i << " :" << y_test << 
                " Prediction: " << y_pred << "\n";
            sum += (y_test == y_pred);
        }

        return static_cast<Scalar>(sum) / static_cast<Scalar>(test_size);
    }

    Vector<Scalar, Dynamic> getLayer(int i){
        return *activations[i];
    }

    friend std::ostream& operator<<(std::ostream& out, const Sequential& model){
        out << "Layer" << '\t' << "Nodes" << '\t' << "Weights" << '\n';
        size_t l = model.weights.size();
        out << 0 << '\t' << model.weights[0]->cols() << '\t' << 0 << '\n';
        for(int i{0}; i < l; i++){
            out << i+1 << '\t' << model.biases[i]->size() << 
            '\t' << model.weights[i]->size() << '\n';
        }
        return out;
    } 
};


#endif