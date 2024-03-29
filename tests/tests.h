#ifndef TEST_H
#define TEST_H

#include <filesystem>
#include "timer.h"
#include "sequential.h"
#include "costs.h"
#include "utils.h"
#include "batchPNGReader.h"
#include "batchCSVReader.h"

namespace fs = std::filesystem;

void testSequentialInit(){
    Sequential2 model({
        new SigmoidLayer(30), 
        new SigmoidLayer(10)
        },
        std::array<Index, 1>{784},
        std::array<Index, 1>{10},
        new MSE()
    );

    size_t n_samples = 10;
    model.init(n_samples);
    std::cout << "Success\n\n";
}

void testFeedFwd(){
    Sequential2 model({
        new SigmoidLayer(30), 
        new SigmoidLayer(10)
        },
        std::array<Index, 1>{2},
        std::array<Index, 1>{10},
        new MSE()
    );

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
        },
        in_shape,
        out_shape,
        new MSE()
    );

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

void testReadBatchPNG(std::string& data_dir) {
    typedef BatchPNGReader::out_data_t data_t;
    typedef BatchPNGReader::out_label_t label_t;
    Index batch_size = 10;
    std::string fullpath{ data_dir + "mnist_png/training" };
    BatchPNGReader batch_reader(fullpath, batch_size);
    batch_reader.reset();
    size_t n_images = batch_reader.size();
    std::cout << "number of training images: " <<
        n_images << "\n\n";
    label_t label_batch;
    data_t image_batch;
    //batch_reader.get(label_batch, image_batch);
    auto begin = batch_reader.begin();
    label_batch = begin.labels();
    image_batch = begin.data();

    Tensor<float, 2> im = image_batch.chip(0, 2);
    Tensor<Index, 0> label = label_batch
        .chip(0, 1).argmax();
    imwrite(im, "./_" + std::to_string(label(0)));

    size_t n = 0;
    auto end = batch_reader.end();
    for(;begin!=end;begin++){
        n++;
    }
    assert(n == (n_images / batch_size));
    begin--;
    label_batch = begin.labels();
    image_batch = begin.data();
    im = image_batch.chip(0, 2);
    label = label_batch
        .chip(0, 1).argmax();
	imwrite(im, "./_" + std::to_string(label(0)));
    std::cout << "Success\n";
}

void testReadBatchCSV(std::string& data_dir) {
    typedef BatchCSVReader::out_data_t data_t;
    typedef BatchCSVReader::out_label_t label_t;
    Index batch_size = 10;
    BatchCSVReader batch_reader(
        data_dir + "mnist_csv/train_x.csv",
        data_dir + "mnist_csv/train_y.csv",
        batch_size
    );
    batch_reader.reset();
    size_t n_images = batch_reader.size();
    std::cout << "number of training images: " <<
        n_images << "\n\n";
    label_t label_batch;
    data_t image_batch;
    auto begin = batch_reader.begin();
    label_batch = begin.labels();
    image_batch = begin.data();

    Tensor<float, 2> im = image_batch.chip(0, 1)
        .reshape(std::array<Index, 2>{28, 28});
    Tensor<Index, 0> label = label_batch
        .chip(0, 1).argmax();
    imwrite(im, "./_" + std::to_string(label(0)));

    size_t n = 0;
    auto end = batch_reader.end();
    for(;begin!=end;begin++){
        n++;
    }
    assert(n == (n_images / batch_size));
    begin--;
    label_batch = begin.labels();
    image_batch = begin.data();
    im = image_batch.chip(0, 1)
        .reshape(std::array<Index, 2>{28, 28});
    label = label_batch
        .chip(0, 1).argmax();
	imwrite(im, "./_" + std::to_string(label(0)));
    std::cout << "Success\n";
}





#endif