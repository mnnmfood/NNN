#ifndef TENSORWRAP_H
#define TENSORWRAP_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include "typedefs.h"

template<typename T, int NumDimensions>
using TensorView = Eigen::TensorMap<Eigen::Tensor<T, NumDimensions>>;

template<typename T>
class TensorWrapper
{
    template<size_t NumDimensions>
    bool checkSize(const std::array<Index, NumDimensions>& size){
        size_t total_size = 1;
        for(size_t n{0}; n < NumDimensions; n++){
            total_size *= size[n];
        }
        return _size == total_size;
    }
public:
    T* data = nullptr;
    size_t _size;

    template<int NumDimensions>
    TensorWrapper(Tensor<T, NumDimensions>& t)
        :data{t.data()}, _size{static_cast<size_t>(t.size())}{}
    template<int NumDimensions>
    TensorWrapper(Tensor<T, NumDimensions>&& t)
        :data{t.data()}, _size{static_cast<size_t>(t.size())}{}
    
    // Return as Eigen 
    template<size_t NumDimensions>
    Eigen::TensorMap<Eigen::Tensor<T, NumDimensions>> get(const std::array<Index, NumDimensions>& size){
        assert(checkSize(size));
        return Eigen::TensorMap<Eigen::Tensor<T, NumDimensions>>(data, size);
    }
    template<size_t NumDimensions>
    Eigen::TensorMap<Eigen::Tensor<T, NumDimensions>> get(std::array<Index, NumDimensions>& size){
        assert(checkSize(size));
        return Eigen::TensorMap<Eigen::Tensor<T, NumDimensions>>(data, size);
    }

    Eigen::TensorMap<Eigen::Tensor<T, 1>> get(){
        return Eigen::TensorMap<Eigen::Tensor<T, 1>>(data, _size);
    }
};

class TensorShape
{
    Index* data;
    size_t _size;
public:
    template<size_t NumDimensions>
    TensorShape(std::array<Index, NumDimensions>& shape) :_size{NumDimensions}{
        data = new Index[_size];
        std::copy(shape.begin(), shape.end(), data);
    }
    template<typename shape_t>
    shape_t get(){
        assert(std::tuple_size<shape_t>{} == _size);
        shape_t temp;
        std::copy(data, data + _size, temp.data());
        return temp;
    }
};


#endif