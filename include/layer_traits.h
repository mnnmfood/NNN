#ifndef FORWARD_H
#define FORWARD_H

#include "typedefs.h"

template<class Derived>
struct traits {};

class FCLayer;
template<> struct traits<FCLayer>
{
    typedef std::array<Index, 1> out_shape_t;
    typedef std::array<Index, 1> in_shape_t;
    const static bool trainable = true;
    const static size_t NumDimensions = 2;
};

template<size_t N> class InputLayer;
template<size_t N> struct traits<InputLayer<N>>
{
    typedef std::array<Index, N> out_shape_t;
    typedef std::array<Index, N> in_shape_t;
    const static bool trainable = false;
    const static size_t NumDimensions {N};
};

template<size_t N> class OutputLayer;
template<size_t N> struct traits<OutputLayer<N>>
{
    typedef std::array<Index, N> out_shape_t;
    typedef std::array<Index, N> in_shape_t;
    const static bool trainable = false;
    const static size_t NumDimensions {N};
};

template<size_t N_in, size_t N_out> class ReshapeLayer;
template<size_t N_in, size_t N_out> struct traits<ReshapeLayer<N_in, N_out>>
{
    typedef std::array<Index, N_out> out_shape_t;
    typedef std::array<Index, N_in> in_shape_t;
    const static bool trainable = false;
    const static size_t NumDimensions {N_out};
};

class FlattenLayer;
template<> struct traits<FlattenLayer>
{
    typedef std::array<Index, 1> out_shape_t;
    typedef std::array<Index, 1> in_shape_t;
    const static bool trainable = false;
    const static size_t NumDimensions = 1;
};

class ConvolLayer;
template<> struct traits<ConvolLayer>
{
    typedef std::array<Index, 4> out_shape_t;
    typedef std::array<Index, 4> in_shape_t;
    const static bool trainable = true;
    const static size_t NumDimensions = 4;
};

class  PoolingLayer;
template<> struct traits<PoolingLayer>
{
    typedef std::array<Index, 4> out_shape_t;
    typedef std::array<Index, 4> in_shape_t;
    const static bool trainable = false;
    const static size_t NumDimensions = 4;
};



#endif
