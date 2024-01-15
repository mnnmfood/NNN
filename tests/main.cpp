
#include <iostream>
#include <iomanip>
#include <vector>

#include <Eigen/Dense>

#include "typedefs.h"
#include "costs.h"
#include "utils.h"
#include "sequential.h"
#include "layers.h"

#include "tests.h"
#include "pngTests.h"

int main(){
    
    std::cout << "--TESTING PNG" << "\n";
    testPNG();

    std::cout << "--TESTING Mnist Data" << "\n";
    std::string dataDir {"../data/mnist_csv/"};
    //#std::vector<Vector<float, Dynamic>*> x;
    Tensor<float, 2> x;
    load_csv(dataDir + "train_x.csv", x);
    std::cout << "Shape x: " << x.dimension(0) << ", " << x.dimension(1)<< '\n';
    Tensor<float, 2> y;
    load_csv(dataDir + "train_y.csv", y);
    std::cout << "Shape y: " << y.dimension(0) << ", " << y.dimension(1)<< '\n';

    Tensor<float, 2> val_x;
    load_csv(dataDir + "val_x.csv", val_x);
    std::cout << "Shape x: " << val_x.dimension(0) << ", " << val_x.dimension(1)<< '\n';

    Tensor<float, 2> val_y;
    load_csv(dataDir + "val_y.csv", val_y);
    std::cout << "Shape y: " << val_y.dimension(0) << ", " << val_y.dimension(1)<< '\n';

    // network architecture
    Sequential2 model({
        new InputLayer(784),
        new SigmoidLayer(30), 
        new SigmoidLayer(10)
        }
        , new MSE());

    model.SGD(x, y, 10, 10, 0.5, 0.00, val_x, val_y);
    //model.GD(x, y, 10, 0.5, 0.01, val_x, val_y);

#if 0
    std::cout << std::setprecision(2);
    std::cout << "Final accuracy "  << model.accuracy(val_x, val_y)*100;
    std::cout << "%" << "\n\n";

    std::cout << "--TESTING INIT" << "\n";
    testSequentialInit();
    std::cout << "--TESTING Feed-forward" << "\n";
    testFeedFwd();
    std::cout << "--TESTING Backwards-propagation" << "\n";
    testBackProp();
#endif
}
