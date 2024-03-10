
#include <iostream>
#include <iomanip>
#include <vector>

#include "typedefs.h"
#include "costs.h"
#include "utils.h"
//#include "sequential.h"
#include "layers.h"

#include "tests.h"
#include "pngTests.h"

#define xstr(x) str(x)
#define str(x) #x

#ifndef DATA_DIR
#define DATA_DIR ../../
#endif

int main(){

    std::cout << " --TESTING PNG" << "\n";
    std::string dataDir = xstr(DATA_DIR);
    testPNG(dataDir);

    std::cout << " --TESTING Batch Images" << "\n";
    testReadBatch(dataDir);

    std::cout << "--TESTING INIT" << "\n";
    testSequentialInit();
    std::cout << "--TESTING Feed-forward" << "\n";
    testFeedFwd();
    std::cout << "--TESTING Backwards-propagation" << "\n";
    testBackProp();
#if 1    
    // model architecture
    Sequential2 model({
        new ReshapeLayer<2, 4>(std::array<Index, 4>{1, 28, 28, 1}),
        new ConvolLayer(std::array<Index, 3>{5, 3, 3}),
        new PoolingLayer(std::array<Index, 2>{3, 3}, 3),
        //new ConvolLayer(std::array<Index, 3>{5, 3, 3}),
        //new PoolingLayer(std::array<Index, 2>{3, 3}, 3),
        new FlattenLayer(),
        new SigmoidLayer(100),
        new SigmoidLayer(10),
        }
        , new CrossEntropy(), 
        std::array<Index, 2>{28, 28},
        std::array<Index, 1>{10});


    
    std::cout << "--TESTING Mnist Data" << "\n";
    std::string trainDir = dataDir + "mnist_png/training";
    std::string testDir = dataDir + "mnist_png/testing";
    int epochs = 10;
    int batch_size = 20;
    float learning_rate = 0.1;
    float momentum = 0.0;
    BatchPNGReader train_reader(trainDir, batch_size);
    BatchPNGReader test_reader(testDir, 100);
    std::cout << "Training Size:" << train_reader.size() << "\n";
    std::cout << "Testing Size:" << test_reader.size() << "\n";
    model.SGD(train_reader, epochs, learning_rate, momentum, test_reader);

    std::cout << std::setprecision(2);
    std::cout << "Final accuracy "  << model.accuracy(test_reader)*100;
    std::cout << "%" << "\n\n";
#endif
#if 0
    std::cout << "--TESTING Mnist Data" << "\n";
    dataDir = dataDir + "/mnist_csv/";
    
    std::array<Eigen::Index, 2> offsets{ 0, 0 };
    std::array<Eigen::Index, 2> extents{ 784, 100 };
    std::array<Eigen::Index, 2> extents_y{ 10, 100 };
    
    int n_samples = -1;
    Tensor<float, 2> x;
    load_csv(dataDir + "train_x.csv", x, n_samples);
    std::cout << "Training Shape x: " << x.dimension(0) << ", " << x.dimension(1)<< '\n';

    Tensor<float, 2> y;
    load_csv(dataDir + "train_y.csv", y, n_samples);
    std::cout << "Training Shape y: " << y.dimension(0) << ", " << y.dimension(1)<< '\n';

    Tensor<float, 2> val_x;
    load_csv(dataDir + "val_x.csv", val_x);
    std::cout << "Validation Shape x: " << val_x.dimension(0) << ", " << val_x.dimension(1)<< '\n';

    Tensor<float, 2> val_y;
    load_csv(dataDir + "val_y.csv", val_y);
    std::cout << "Validation Shape y: " << val_y.dimension(0) << ", " << val_y.dimension(1)<< '\n';
    // network architecture
    Sequential2 model2({
        new ReshapeLayer<1, 4>(std::array<Index, 4>{1, 28, 28, 1}),
        new ConvolLayer(std::array<Index, 3>{10, 3, 3}),
        new PoolingLayer(std::array<Index, 2>{3, 3}, 1),
        new ConvolLayer(std::array<Index, 3>{5, 3, 3}),
        new PoolingLayer(std::array<Index, 2>{3, 3}, 1),
        new FlattenLayer(),
        //new SigmoidLayer(100),
        new SigmoidLayer(10),
        }
        , new MSE(), 
        std::array<Index, 1>{784},
        std::array<Index, 1>{10});

    // Optimize using Stochastic Gradient Descent
    int epochs = 2;
    int batch_size = 20;
    float learning_rate = 0.01;
    float momentum = 0.0;
    model2.SGD(x, y, epochs, batch_size, learning_rate, momentum, val_x, val_y);

    std::cout << std::setprecision(2);
    std::cout << "Final accuracy "  << model2.accuracy(val_x, val_y)*100;
    std::cout << "%" << "\n\n";
#endif
}
