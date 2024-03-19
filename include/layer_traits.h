#ifndef FORWARD_H
#define FORWARD_H

#include <string_view>
#include "typedefs.h"

template<class Derived>
struct traits {};

// Layer traits
class FCLayer;
template<> struct traits<FCLayer>
{
    typedef std::array<Index, 1> out_shape_t;
    typedef std::array<Index, 1> in_shape_t;
    const static bool trainable = true;
    const static size_t NumDimensions = 2;
    inline constexpr static std::string_view description = "Fully Connected Layer";
};

template<size_t N> class InputLayer;
template<size_t N> struct traits<InputLayer<N>>
{
    typedef std::array<Index, N> out_shape_t;
    typedef std::array<Index, N> in_shape_t;
    const static bool trainable = false;
    const static size_t NumDimensions {N};
    inline constexpr static std::string_view description = "Input Layer";
};

template<size_t N> class OutputLayer;
template<size_t N> struct traits<OutputLayer<N>>
{
    typedef std::array<Index, N> out_shape_t;
    typedef std::array<Index, N> in_shape_t;
    const static bool trainable = false;
    const static size_t NumDimensions {N};
    inline constexpr static std::string_view description = "Output Layer";
};

template<size_t N_in, size_t N_out> class ReshapeLayer;
template<size_t N_in, size_t N_out> struct traits<ReshapeLayer<N_in, N_out>>
{
    typedef std::array<Index, N_out> out_shape_t;
    typedef std::array<Index, N_in> in_shape_t;
    const static bool trainable = false;
    const static size_t NumDimensions {N_out};
    inline constexpr static std::string_view description = "Reshape Layer";
};

class FlattenLayer;
template<> struct traits<FlattenLayer>
{
    typedef std::array<Index, 1> out_shape_t;
    typedef std::array<Index, 1> in_shape_t;
    const static bool trainable = false;
    const static size_t NumDimensions = 1;
    inline constexpr static std::string_view description = "Flatten Layer";
};

class ConvolLayer;
template<> struct traits<ConvolLayer>
{
    typedef std::array<Index, 4> out_shape_t;
    typedef std::array<Index, 4> in_shape_t;
    const static bool trainable = true;
    const static size_t NumDimensions = 4;
    inline constexpr static std::string_view description = "Convolutional Layer";
};

class  PoolingLayer;
template<> struct traits<PoolingLayer>
{
    typedef std::array<Index, 4> out_shape_t;
    typedef std::array<Index, 4> in_shape_t;
    const static bool trainable = false;
    const static size_t NumDimensions = 4;
    inline constexpr static std::string_view description = "Pooling Layer";
};

// Cost Function traits
class MSE;
template<> struct traits <MSE>
{
    typedef std::array<Index, 2> shape_t;
};

class CrossEntropy;
template<> struct traits <CrossEntropy>
{
    typedef std::array<Index, 2> shape_t;
};

template<size_t N> class DummyCost;
template<size_t N> struct traits <DummyCost<N>>
{
    typedef std::array<Index, N + 1> shape_t;
};

// Reader traits
class BatchPNGReader;
struct BatchPNGIterator;
template<> struct traits <BatchPNGReader>
{
    typedef BatchPNGIterator iterator;
    typedef std::pair<int, std::string> data_t;
	typedef Tensor<float, 3> out_data_t;
	typedef Tensor<float, 2> out_label_t;
};

class BatchCSVReader;
struct BatchCSVIterator;
template<> struct traits <BatchCSVReader>
{
    typedef BatchCSVIterator iterator;
    typedef std::pair<
        std::pair<Index, Index>,
		std::pair<Index, Index>> data_t;
    typedef Tensor<float, 2> out_data_t;
    typedef Tensor<float, 2> out_label_t;
};




#endif
