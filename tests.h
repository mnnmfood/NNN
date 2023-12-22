#ifndef TEST_H
#define TEST_H

#include "sequential2.h"
#include "activations.h"

void printShape(Matrix<double, Dynamic, Dynamic>* a){ 
    std::cout << " Shape: (" << a->rows() << ", " << a->cols() << ")\n";
}

void testSequentialInit(){
    std::vector<int> arch {1, 2, 3, 4};
    Sequential<double> model(arch, new Logistic<double>());
    size_t n_samples = 10;
    model.initSGD(n_samples);
    std::cout << "Delta layer1, ";
    printShape(model.delta.front());
    std::cout << "Delta layer-1, ";
    printShape(model.delta.back());
}

void testFeedFwd(){
    std::vector<int> arch {2, 3, 3, 2};
    Sequential<double> model(arch, new Logistic<double>());
    Eigen::MatrixXd x{{1, 1}, {1, 1}, {1, 1}};
    x.transposeInPlace();

    model.initSGD(x.cols());

    for(int i{0}; i < model.num_layers-1; i++){
        model.weights[i]->setConstant(0.5);
        model.biases[i]->setConstant(0.5);
    }
    std::cout << "Input layer ";
    printShape(model.activations[0]);
    std::cout << "Input ";
    printShape(&x);

    model.feedFwd(x);
    std::cout << "Input: " << x << "\n\n";
    std::cout << "Output: " << *model.activations.back() << "\n\n";
}

#endif