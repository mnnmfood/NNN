
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

    std::vector<Vector<double, Dynamic>*> x;
    std::string fpath{"train_x.csv"};
    load_csv(fpath, x);
    //Vector<double, Dynamic> x0 {{2734, 342, 24, 23, 1, 90}};
    //x.push_back(&x0);
    std::cout << "Size x: " << x.front()->size() << '\n';

    std::vector<Vector<double, Dynamic>*> y;
    fpath = "train_y.csv";
    load_csv(fpath, y);
    //Vector<double, Dynamic> y0 {{234, 53, 12}};
    //y.push_back(&y0);
    std::cout << "Size y: " << y.front()->size() << '\n';

    std::vector<int> arch {784, 256, 10};
    Sequential<double> model(arch, new Logistic<double>());
    std::cout << model << '\n';
    model.SGD(x, y, 200, x.size(), 0.2);

    std::cout << std::setprecision(2);
    std::cout << "Accuracy"  << model.accuracy(x, y);

#if 0
    Sequential<double> model(arch, new Logistic<double>());
    std::cout << model;
    Eigen::Vector3d input{5, 5, 5};
    model.feedFwd(input);
    Eigen::Vector<double, Dynamic> layer {model.getLayer(0)};
    std::cout << layer << '\n';
#endif
}
