
#include <iostream>
#include <iomanip>
#include <vector>

#include "typedefs.h"
#include "costs.h"
#include "utils.h"
#include "sequential.h"
#include "layers.h"

#include "tests.h"
#include "pngTests.h"

int main(){
#if 1
    std::cout << " --TESTING PNG" << "\n";
    //testPNG();

    std::cout << "--TESTING INIT" << "\n";
    testSequentialInit();
    std::cout << "--TESTING Feed-forward" << "\n";
    testFeedFwd();
    std::cout << "--TESTING Backwards-propagation" << "\n";
    testBackProp();

    std::cout << "--TESTING Mnist Data" << "\n";
    std::string dataDir {"../../../data/mnist_csv/"};
    std::cout << "HEEAPPPP\n\n";
    
    std::array<Eigen::Index, 2> offsets{ 0, 0 };
    std::array<Eigen::Index, 2> extents{ 784, 100 };
    std::array<Eigen::Index, 2> extents_y{ 10, 100 };

    Tensor<float, 2> x;
    load_csv(dataDir + "train_x.csv", x);
    //x = x.slice(offsets, extents);
    std::cout << "Training Shape x: " << x.dimension(0) << ", " << x.dimension(1)<< '\n';

    Tensor<float, 2> y;
    load_csv(dataDir + "train_y.csv", y);
    //y = y.slice(offsets, extents_y);
    std::cout << "Training Shape y: " << y.dimension(0) << ", " << y.dimension(1)<< '\n';

    Tensor<float, 2> val_x;
    load_csv(dataDir + "val_x.csv", val_x);
    std::cout << "Validation Shape x: " << val_x.dimension(0) << ", " << val_x.dimension(1)<< '\n';

    Tensor<float, 2> val_y;
    load_csv(dataDir + "val_y.csv", val_y);
    std::cout << "Validation Shape y: " << val_y.dimension(0) << ", " << val_y.dimension(1)<< '\n';

    // network architecture
    Sequential2 model({
        new ReshapeLayer<1, 4>(std::array<Index, 4>{1, 28, 28, 1}),
        new ConvolLayer(std::array<Index, 3>{32, 3, 3}),
        new PoolingLayer(std::array<Index, 2>{2, 2}, 2),
        new FlattenLayer(),
        new SigmoidLayer(100),
        new SigmoidLayer(10),
        }
        , new MSE(), 
        std::array<Index, 1>{784},
        std::array<Index, 1>{10});

    // Optimize using Stochastic Gradient Descent
    int epochs = 1;// 20;
    int batch_size = 20;
    std::cout << x.dimensions();
    float learning_rate = 0.01;
    float momentum = 0.0;
    model.SGD(x, y, epochs, batch_size, learning_rate, momentum, val_x, val_y);

    std::cout << std::setprecision(2);
    std::cout << "Final accuracy "  << model.accuracy(val_x, val_y)*100;
    std::cout << "%" << "\n\n";
#endif
}
