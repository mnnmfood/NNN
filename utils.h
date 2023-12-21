#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <vector>

using Eigen::Matrix;
using Eigen::Vector;
using Eigen::Dynamic;

int load_csv(std::string fpath, 
                std::vector<Vector<double, Dynamic>*>& x_data); 

#endif