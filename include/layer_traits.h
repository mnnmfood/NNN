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

#if 0
class ConvolLayer;
template<> struct traits<ConvolLayer>
{
    typedef std::array<Index, 3> out_shape_t;
    typedef std::array<Index, 3> in_shape_t;
    static bool trainable;
    static size_t NumDimensions;
};
bool traits<ConvolLayer>::trainable = true;
size_t traits<ConvolLayer>::NumDimensions = 3;
#endif

#endif
