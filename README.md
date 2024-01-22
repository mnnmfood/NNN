# NNN
## Description
Novice Neural Networks:

Neural networks built from scratch using Eigen Tensor module. An example of its use can be found in the file **tests/main.cpp**. This repo also contains a custom made wrapper around [libpng](http://www.libpng.org/).
#### Types of layers implemented (in the process of adding convolutional and max-pooling layers):
 - Sigmoid
 - Tanh
 - Softmax
#### Cost functions:
  - Mean-Squared Error  
  - Cross-entropy
## Requirements:
  - Eigen 3.4.0
  - libpng 1.2.56  
  - zlib 1.3
## How to build the examples
Make sure the required static libraries (libpng.a and libz.a) as well as the header files (eigen-3.4.0/ and png.h) are in the PATH.
```
cd build
cmake ../
cmake --build .
./tests
```
