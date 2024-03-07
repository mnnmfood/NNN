#ifndef TEST_H
#define TEST_H

#include <filesystem>
#include "timer.h"
#include "sequential.h"
#include "costs.h"
#include "utils.h"
#include "batchReader.h"

namespace fs = std::filesystem;

void printShape(const Matrix<float, Dynamic, Dynamic>& a){ 
    std::cout << " Shape: (" << a.rows() << ", " << a.cols() << ")\n";
}

void testSequentialInit(){
    Sequential2 model({
        new SigmoidLayer(30), 
        new SigmoidLayer(10)
        }
        , new MSE(),
        std::array<Index, 1>{784},
        std::array<Index, 1>{10});

    size_t n_samples = 10;
    model.init(n_samples);
    std::cout << "Success\n\n";
}

void testFeedFwd(){
    Sequential2 model({
        new SigmoidLayer(30), 
        new SigmoidLayer(10)
        }
        , new MSE(),
        std::array<Index, 1>{2},
        std::array<Index, 1>{10});

    Eigen::Tensor<float, 2> x(2, 10);
    //Eigen::VectorXf xi {{5, 3}};
    Eigen::Tensor<float, 1> xi(2);
    xi.setValues({5, 3});
    for(int i{0}; i < 10; i++){x.chip(i, 1) = xi;}
    model.init(x.dimension(1));
    model.fwdProp(x);
    std::cout << "Success\n\n";
}

void testBackProp(){
    std::array<Index, 1> in_shape{ 36 };
    std::array<Index, 1> out_shape{ 8 };
    Sequential2 model({
        new ReshapeLayer<1, 4>(std::array<Index, 4>({1, 6, 6, 1})),
        new ConvolLayer(std::array<Index, 3>({2, 3, 3})),
        new PoolingLayer(std::array<Index, 2>({2, 2}), 1),
        new FlattenLayer(),
        new SigmoidLayer(8)
        }
        , new MSE(),
        in_shape,
        out_shape);

    int n_samples {2};

    Eigen::Tensor<float, 2> x(in_shape[0], n_samples);
    x.setRandom();

    model.init(n_samples);
    model.fwdProp(x);
    
    Eigen::Tensor<float, 2> y(out_shape[0], n_samples);
    y.setRandom();

    model.bkwProp(y);
    std::cout << "Success\n";
}

void testReadBatch(std::string& data_dir) {
    Index batch_size = 10;
    std::string fullpath{ data_dir + "mnist_png/testing/0" };
    BatchPNGReader batch_reader(fullpath, batch_size);

    Tensor<byte, 3> im_batch = batch_reader.get();
    imwrite(im_batch.chip(0, 2), "./batch_test1");

    batch_reader++;
    im_batch = batch_reader.get();
    imwrite(im_batch.chip(0, 2), "./batch_test2");
    std::cout << "Success\n";
}



#endif