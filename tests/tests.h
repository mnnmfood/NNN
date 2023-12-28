#ifndef TEST_H
#define TEST_H

#include "sequential.h"
#include "activations.h"
#include "costs.h"

void printShape(Matrix<double, Dynamic, Dynamic>& a){ 
    std::cout << " Shape: (" << a.rows() << ", " << a.cols() << ")\n";
}

void testSequentialInit(){
    std::vector<int> arch {1, 2, 3, 4};
    Logistic<double> log;
    MSE<double> mse;
    Sequential<double> model(arch, log, mse, false);
    size_t n_samples = 10;
    model.initGD(n_samples);
    std::cout << "Delta layer1, ";
    printShape(model.delta.front());
    std::cout << "Delta layer-1, ";
    printShape(model.delta.back());
    std::cout << "\n\n";
}

void testFeedFwd(){
    std::vector<int> arch {2, 3, 3, 2};
    Logistic<double> log;
    MSE<double> mse;
    Sequential<double> model(arch, log, mse, false);
    size_t n_samples = 10;
    Eigen::MatrixXd x(2, 10);
    Eigen::VectorXd xi {{5, 3}};
    for(int i{0}; i < 10; i++){x.col(i) = xi;}

    model.initGD(x.cols());

    std::cout << "Input layer ";
    printShape(model.activations[0]);
    std::cout << "Input ";
    printShape(x);

    model.feedFwd(x);
    std::cout << "Input: \n" << x << "\n\n";
    std::cout << "Output: \n" << model.activations.back() << "\n\n";
}

void testBackProp(){
    std::vector<int> arch {6, 3, 3, 3};

    Logistic<double> log;
    MSE<double> mse;
    Sequential<double> model(arch, log, mse, false);
    int n_samples {2};
    Eigen::VectorXd xi {{2734, 342, 24, 23, 1, 90}};
    Eigen::MatrixXd x(6, n_samples);
    for(int i{0}; i < n_samples; i++){x.col(i) = xi;}

    model.initGD(n_samples);
    model.feedFwd(x);

    Eigen::VectorXd yi {{234, 53, 12}};
    Eigen::MatrixXd y(3, n_samples);
    for(int i{0}; i < n_samples; i++){y.col(i) = yi;}

    std::cout << "Output: \n" << y << "\n\n";
    std::cout << "Prediction: \n" << model.activations.back() << "\n\n";

    model.backProp(x, y);
    for(int i{0}; i < model.num_layers-1; i++){
        model.weights[i] -= (0.9 / n_samples) * (model.nabla_w[i]);
        model.biases[i] -= (0.9 / n_samples) * (model.nabla_b[i]);
    }

    model.feedFwd(x);

    std::cout << "Output: \n" << y << "\n\n";
    std::cout << "Prediction: \n" << model.activations.back() << "\n\n";
}

#endif