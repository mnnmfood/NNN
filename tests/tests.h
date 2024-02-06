#ifndef TEST_H
#define TEST_H

#include "sequential.h"
#include "costs.h"

void printShape(const Matrix<float, Dynamic, Dynamic>& a){ 
    std::cout << " Shape: (" << a.rows() << ", " << a.cols() << ")\n";
}

void testSequentialInit(){
    Sequential2 model({
        new SigmoidLayer(30), 
        new SigmoidLayer(10)
        }
        , new MSE(),
        std::array<Index, 1>{784},
        std::array<Index, 1>{10});

    size_t n_samples = 10;
    model.init(n_samples);
    std::cout << "Success\n\n";
}

void testFeedFwd(){
    Sequential2 model({
        new SigmoidLayer(30), 
        new SigmoidLayer(10)
        }
        , new MSE(),
        std::array<Index, 1>{2},
        std::array<Index, 1>{10});

    Eigen::Tensor<float, 2> x(2, 10);
    //Eigen::VectorXf xi {{5, 3}};
    Eigen::Tensor<float, 1> xi(2);
    xi.setValues({5, 3});
    for(int i{0}; i < 10; i++){x.chip(i, 1) = xi;}
    model.init(x.dimension(1));
    model.fwdProp(x);
    std::cout << "Success\n\n";
}

void testBackProp(){
    Sequential2 model({
        new ReshapeLayer<1, 3>(std::array<Index, 3>({6,6,1})),
        new ConvolLayer(std::array<Index, 3>({3, 3, 2})),
        new ReshapeLayer<3, 1>(std::array<Index, 1>({32})),
        new SigmoidLayer(32)
        }
        , new MSE(),
        std::array<Index, 1>{36},
        std::array<Index, 1>{32});

    int n_samples {2};

    Eigen::Tensor<float, 1> xi(36);
    Eigen::Tensor<float, 2> x(36, n_samples);
    for(int i{0}; i < n_samples; i++){
        xi.setConstant(i);
        x.chip(i, 1) = xi;
    }

    model.init(n_samples);
    model.fwdProp(x);

    Eigen::Tensor<float, 1> yi(32); 
    Eigen::Tensor<float, 2> y(32, n_samples);
    for(int i{0}; i < n_samples; i++){
        y.setConstant(i);
        y.chip(i, 1) = yi;
    }

    model.bkwProp(y);
    std::cout << "Success\n";
}

#endif