#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <vector>

using Eigen::Matrix;
using Eigen::Vector;
using Eigen::Dynamic;

int load_csv(std::string fpath, Matrix<double, Dynamic, Dynamic>& data);

#endif