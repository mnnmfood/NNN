
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

    //#std::vector<Vector<float, Dynamic>*> x;
    Matrix<float, Dynamic, Dynamic> x;
    load_csv("train_x.csv", x);
    std::cout << "Shape x: " << x.rows() << ", " << x.cols()<< '\n';

    Matrix<float, Dynamic, Dynamic> y;
    load_csv("train_y.csv", y);
    std::cout << "Shape y: " << y.rows() << ", " << y.cols()<< '\n';
    std::cout << y(all, Eigen::seqN(0, 10));

    Matrix<float, Dynamic, Dynamic> val_x;
    load_csv("val_x.csv", val_x);
    std::cout << "Shape x: " << val_x.rows() << ", " << val_x.cols()<< '\n';

    Matrix<float, Dynamic, Dynamic> val_y;
    load_csv("val_y.csv", val_y);
    std::cout << "Shape y: " << val_y.rows() << ", " << val_y.cols()<< '\n';

    // network architecture
    std::vector<int> arch {784, 30, 10};
    Sequential<float> model(arch, new Logistic<float>(), new CrossEntropy<float>());
    std::cout << model << '\n';

    model.SGD(x, y, 100, 10, 0.5, val_x, val_y);
    //model.GD(x, y, 10, 0.5, val_x, val_y);
    //model.GD(x, y, 30, 0.5);

    std::cout << std::setprecision(2);
    std::cout << "Accuracy "  << model.accuracy(val_x, val_y) << "\n\n";

#ifdef TESTING
    std::cout << "TESTING" << "\n\n";
    testSequentialInit();
    testFeedFwd();
    testBackProp();
#endif
}
