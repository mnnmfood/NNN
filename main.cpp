
#include <iostream>
#include <iomanip>
#include <vector>
#include <Eigen/Dense>
#include "activations.h"
#include "utils.h"
#include "sequential.h"
#include "tests.h"

using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::Vector;

int main(){

    //#std::vector<Vector<double, Dynamic>*> x;
    Matrix<double, Dynamic, Dynamic> x;
    std::string fpath{"train_x.csv"};
    load_csv(fpath, x);
    std::cout << "Shape x: " << x.rows() << ", " << x.cols()<< '\n';

    Matrix<double, Dynamic, Dynamic> y;
    fpath = "train_y.csv";
    load_csv(fpath, y);
    std::cout << "Shape x: " << y.rows() << ", " << y.cols()<< '\n';

    // network architecture
    std::vector<int> arch {784, 128, 64, 10};
    Sequential<double> model(arch, new Logistic<double>());
    std::cout << model << '\n';

    //model.SGD(x, y, 200, 100, 0.1);
    model.GD(x, y, 200, 0.9);

    std::cout << std::setprecision(2);
    std::cout << "Accuracy"  << model.accuracy(x, y) << "\n\n";

#ifdef TESTING
    std::cout << "TESTING" << "\n\n";
    testSequentialInit();
    testFeedFwd();
    testBackProp();
#endif
}
