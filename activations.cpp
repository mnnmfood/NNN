#include <cmath>
#include "activation.h"

double logistic(double x, double b, double w){
    return 1 / (1 + std::exp(b + w*x));
}