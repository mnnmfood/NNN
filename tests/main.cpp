
#include <iostream>
#include <iomanip>
#include <vector>

#include "typedefs.h"
#include "costs.h"
#include "utils.h"
#include "sequential.h"
#include "layers.h"
#include "batchCSVReader.h"
#include "batchPNGReader.h"

#include "tests.h"
#include "pngTests.h"
#include "testOps.h"

#define xstr(x) str(x)
#define str(x) #x

#ifndef DATA_DIR
#define DATA_DIR ../../
#endif

int main() {
    std::string dataDir = xstr(DATA_DIR);

#if 1
    std::cout << " --TESTING PNG" << "\n";
    testPNG(dataDir);

    std::cout << " --TESTING Batch Images" << "\n";
    testReadBatchPNG(dataDir);
    testReadBatchCSV(dataDir);

    std::cout << "--TESTING INIT" << "\n";
    testSequentialInit();
    std::cout << "--TESTING Feed-forward" << "\n";
    testFeedFwd();
    std::cout << "--TESTING Backwards-propagation" << "\n";
    testBackProp();
    std::cout << "--TESTING Convolution Ops" << "\n";
    testAllOps();

#endif
    // model architecture
    Sequential2 model({
        //new ReshapeLayer<1, 4>(std::array<Index, 4>{1, 28, 28, 1}),
        //new ConvolLayer(std::array<Index, 3>{5, 3, 3}),
        //new PoolingLayer(std::array<Index, 2>{3, 3}, 3),
        new FlattenLayer(),
        new SigmoidLayer(128),
        new SigmoidLayer(10),
        },
        std::array<Index, 1>{784},
        std::array<Index, 1>{10},
        new CrossEntropy()
    );
    std::cout << "--TESTING Mnist Data" << "\n";
    int epochs = 10;
    int batch_size = 20;
    float learning_rate = 0.1;
    float momentum = 0.0;
    std::string trainDir = dataDir + "mnist_png/training";
    std::string testDir = dataDir + "mnist_png/testing";
    //BatchPNGReader train_reader(trainDir, batch_size);
    //BatchPNGReader test_reader(testDir, 100);
    BatchCSVReader train_reader(
        dataDir + "mnist_csv/train_x.csv", 
        dataDir + "mnist_csv/train_y.csv", 
        batch_size);
    BatchCSVReader test_reader(
        dataDir + "mnist_csv/val_x.csv", 
        dataDir + "mnist_csv/val_y.csv", 
        100);
    std::cout << "Training Size:" << train_reader.size() << "\n";
    std::cout << "Testing Size:" << test_reader.size() << "\n";
    model.SGD(train_reader, epochs, learning_rate, momentum, test_reader);

    std::cout << std::setprecision(2);
    std::cout << "Final accuracy " << model.accuracy(test_reader) * 100;
    std::cout << "%" << "\n\n";
}
