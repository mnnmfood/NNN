#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#define EIGEN_USE_THREADS

#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/ThreadPool>

using Eigen::Index;
using Eigen::Tensor;
using Eigen::TensorMap;
using Eigen::TensorRef; 
using Eigen::ThreadPoolDevice;
using Eigen::ThreadPool;

using byte = unsigned char;

enum ConvolTypes{
    valid,
    full,
};

#endif