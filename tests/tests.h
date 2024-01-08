#ifndef TEST_H
#define TEST_H

#include "sequential.h"
#include "costs.h"

void printShape(const Matrix<float, Dynamic, Dynamic>& a){ 
    std::cout << " Shape: (" << a.rows() << ", " << a.cols() << ")\n";
}

void testSequentialInit(){
    Sequential2 model({
        new InputLayer(784),
        new SigmoidLayer(30), 
        new SigmoidLayer(10)
        }
        , new MSE());

    size_t n_samples = 10;
    model.init(n_samples);
    std::cout << "Success\n\n";
}

void testFeedFwd(){
    Sequential2 model({
        new InputLayer(2),
        new SigmoidLayer(30), 
        new SigmoidLayer(10)
        }
        , new MSE());

    Eigen::MatrixXf x(2, 10);
    Eigen::VectorXf xi {{5, 3}};
    for(int i{0}; i < 10; i++){x.col(i) = xi;}

    model.init(x.cols());
    model.fwdProp(x);
    std::cout << "Success\n\n";
}

void testBackProp(){
    Sequential2 model({
        new InputLayer(6),
        new SigmoidLayer(30), 
        new SigmoidLayer(3)
        }
        , new MSE());

    int n_samples {2};
    Eigen::VectorXf xi {{2734, 342, 24, 23, 1, 90}};
    Eigen::MatrixXf x(6, n_samples);
    for(int i{0}; i < n_samples; i++){x.col(i) = xi;}

    model.init(n_samples);
    model.fwdProp(x);

    Eigen::VectorXf yi {{234, 53, 12}};
    Eigen::MatrixXf y(3, n_samples);
    for(int i{0}; i < n_samples; i++){y.col(i) = yi;}

    model.bkwProp(y);
    std::cout << "Success\n";
}

#endif