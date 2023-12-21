
#include <iostream>
#include <iomanip>
#include <vector>
#include <Eigen/Dense>
#include "activations.h"
#include "utils.h"
#include "sequential.h"

using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::Vector;

int main(){
    Logistic<double> a;
    Matrix<double, 10, 1> test {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::cout << a.activation(test.array()) << '\n';

    std::vector<Vector<double, Dynamic>*> x;
    std::string fpath{"train_x.csv"};
    //load_csv(fpath, x);
    Vector<double, Dynamic> x0 {{2734, 342, 24, 23, 1, 90}};
    x.push_back(&x0);
    std::cout << "Size x: " << x.front()->size() << '\n';

    std::vector<Vector<double, Dynamic>*> y;
    fpath = "train_y.csv";
    //load_csv(fpath, y);
    Vector<double, Dynamic> y0 {{234, 53, 12}};
    y.push_back(&y0);
    std::cout << "Size y: " << y.front()->size() << '\n';

    std::cout << *y[0] << '\n';
    std::cout << *y.back() * (*y.back()).transpose() << '\n';

    std::vector<int> arch {6, 3, 3, 3};
    Sequential<double> model(arch, new Logistic<double>());
    std::cout << model << '\n';
    model.updateBatch(x, y, 0.1);

    std::cout << std::setprecision(2);
    std::cout << "result" << '\n';
    std::cout << *model.weights[0] << "\n\n";
    std::cout << *model.weights[1] << "\n\n";
    std::cout << *model.weights[2] << "\n\n";
    std::cout << *model.biases[0] << "\n\n";
    std::cout << *model.biases[1] << "\n\n";
    std::cout << *model.biases[2] << "\n\n";

#if 0
    Sequential<double> model(arch, new Logistic<double>());
    std::cout << model;
    Eigen::Vector3d input{5, 5, 5};
    model.feedFwd(input);
    Eigen::Vector<double, Dynamic> layer {model.getLayer(0)};
    std::cout << layer << '\n';
#endif
}
