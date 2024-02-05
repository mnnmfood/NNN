#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::Index;
using Eigen::Tensor;
using Eigen::TensorMap;

using byte = unsigned char;

struct dummyTensor{};

enum ConvolTypes{
    valid,
    full,
};

#endif