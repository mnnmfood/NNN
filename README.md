# NNN
## Description
Novice Neural Networks:

Neural networks built from scratch using Eigen Tensor module. An example of its use can be found in the file **tests/main.cpp**. 

This repo also contains a custom made wrapper around [libpng](http://www.libpng.org/). An example of its use can be found in the file  **tests/pngTests.h**
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
  - Boost 

## Building
```
cd build
cmake ../
cmake --build .
./tests
```
The result of the build is two static libraries: src.lib and pngwrapper.lib. Before building make sure that the required packages (eigen-3.4.0/, png.h, boost/) 
as well as the required static libraries (libpng.lib and zlib.lib) are in the PATH.
