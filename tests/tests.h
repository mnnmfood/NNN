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
    std::cout << "Success\n\n";
}

void testReadBatch(std::string& data_dir) {
    Index batch_size = 10;
    std::string fullpath{ data_dir + "mnist_png/training" };
    BatchPNGReader batch_reader(fullpath, batch_size);
    size_t n_images = batch_reader.size();
    std::cout << "number of training images: " <<
        n_images << "\n\n";
    Tensor<float, 2> label_batch;
    Tensor<byte, 3> image_batch;
    //batch_reader.get(label_batch, image_batch);
    auto begin = batch_reader.begin();
    label_batch = begin.labels();
    image_batch = begin.images();

    Tensor<byte, 2> im = image_batch.chip(0, 2);
    Tensor<Index, 0> label = label_batch
        .chip(0, 1).argmax();
    imwrite(im, "./_" + std::to_string(label(0)));

    size_t n = 0;
    auto end = batch_reader.end();
    for(;begin!=end;begin++){
        n++;
        if (n == 98) {
            int a = 9;
        }
    }
    assert(n == (n_images / batch_size));
    begin--;
    label_batch = begin.labels();
    image_batch = begin.images();
    im = image_batch.chip(0, 2);
    label = label_batch
        .chip(0, 1).argmax();
	imwrite(im, "./_" + std::to_string(label(0)));
    std::cout << "Success\n";
}



#endif