
#include <iostream>
#include <iomanip>
#include <vector>
#include <Eigen/Dense>
#include "activations.h"
#include "costs.h"
#include "utils.h"
#include "sequential.h"
#include "tests.h"

using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::Vector;

int main(){
#ifdef FLAGSET
    std::cout << "Flags set\n\n";
#endif
    std::cout << "--TESTING Mnist Data" << "\n";
    std::string dataDir {"../data/mnist/"};
    //#std::vector<Vector<float, Dynamic>*> x;
    Matrix<float, Dynamic, Dynamic> x;
    load_csv(dataDir + "train_x.csv", x);
    std::cout << "Shape x: " << x.rows() << ", " << x.cols()<< '\n';

    Matrix<float, Dynamic, Dynamic> y;
    load_csv(dataDir + "train_y.csv", y);
    std::cout << "Shape y: " << y.rows() << ", " << y.cols()<< '\n';

    Matrix<float, Dynamic, Dynamic> val_x;
    load_csv(dataDir + "val_x.csv", val_x);
    std::cout << "Shape x: " << val_x.rows() << ", " << val_x.cols()<< '\n';

    Matrix<float, Dynamic, Dynamic> val_y;
    load_csv(dataDir + "val_y.csv", val_y);
    std::cout << "Shape y: " << val_y.rows() << ", " << val_y.cols()<< '\n';

    // network architecture
    std::vector<int> arch {784, 30, 30, 10};
    Logistic<float> activ;
    MSE<float> cost;
    Sequential<float> model(arch, activ, cost, true);
    //Sequential<float> model(arch, new Tanh<float>(), new MSE<float>());
    std::cout << model << '\n';

    model.SGD(x, y, 10, 10, 0.5, 0.00, val_x, val_y);
    //model.GD(x, y, 10, 0.5, 0.01, val_x, val_y);

    std::cout << std::setprecision(2);
    std::cout << "Final accuracy "  << model.accuracy(val_x, val_y)*100;
    std::cout << "%" << "\n\n";

    std::cout << "--TESTING INIT" << "\n";
    testSequentialInit();
    std::cout << "--TESTING Feed-forward" << "\n";
    testFeedFwd();
    std::cout << "--TESTING Backwards-propagation" << "\n";
    testBackProp();
}
