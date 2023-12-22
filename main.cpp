
#include <iostream>
#include <iomanip>
#include <vector>
#include <Eigen/Dense>
#include "activations.h"
#include "utils.h"
#include "sequential2.h"
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
    std::vector<int> arch {784, 256, 10};
    Sequential<double> model(arch, new Logistic<double>());
    std::cout << model << '\n';

#if 0
    model.SGD(x, y, 200, x.size(), 0.2);

    std::cout << std::setprecision(2);
    std::cout << "Accuracy"  << model.accuracy(x, y);
#endif

#ifdef TESTING
    std::cout << "TESTING" << "\n\n";
    testSequentialInit();
    testFeedFwd();
#endif
}
