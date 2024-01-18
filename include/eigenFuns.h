#ifndef FUNCTORS_H
#define FUNCTORS_H

#include <functional>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "typedefs.h"

// ---- Sum Matrix and Vector colwise or rowwise
template<typename T>
struct VecsumOp
{
  VecsumOp(Tensor<T, 2>& mat, Tensor<T, 1>& vec, bool rowwise=true) :m_mat{mat}, m_vec{vec} {
    if(rowwise){
      assert(m_mat.dimension(1)==m_vec.dimension(0));
    }else{
      assert(m_mat.dimension(0)==m_vec.dimension(0));
    }
  } 
  const T operator()(Eigen::Index row, Eigen::Index col) const{
    return m_mat(row, col) + (m_rowwise?m_vec(col):m_vec(row));
  }
private:
  Tensor<T, 2> m_mat;
  Tensor<T, 1> m_vec;
  bool m_rowwise;
};
// R-value version
template<typename T>
Tensor<T, 2> vecSum(Tensor<T, 2>&& mat, Tensor<T, 1>& vec, bool rowwise=true){
    return mat.nullaryExpr(VecsumOp(mat, vec, rowwise));
}
// L-value version
template<typename T>
Tensor<T, 2> vecSum(Tensor<T, 2>& mat, Tensor<T, 1>& vec, bool rowwise=true){
    return mat.nullaryExpr(VecsumOp(mat, vec, rowwise));
}
// ---

// --- Reducer: calculate squared norm of either rows or cols
template <typename T> 
struct SqNormReducer
{
   EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) const {
     Eigen::internal::scalar_sum_op<T> sum_op;
     *accum = sum_op(*accum, t);
   }
   template <typename Packet>
   EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reducePacket(const Packet& p, Packet* accum) const {
     (*accum) = Eigen::internal::padd<Packet>(*accum, p);
   }
  
   EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
     Eigen::internal::scalar_cast_op<int, T> conv;
     return conv(0);
   }
   template <typename Packet>
   EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet initializePacket() const {
     return Eigen::internal::pset1<Packet>(initialize());
   }
   EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T accum) const {
     Eigen::internal::scalar_sqrt_op<T> sqrt_op;
     return sqrt_op(accum);
   }
   template <typename Packet>
   EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet finalizePacket(const Packet& vaccum) const {
     return vaccum;
   }
   template <typename Packet>
   EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalizeBoth(const T saccum, const Packet& vaccum) const {
     Eigen::internal::scalar_sum_op<T> sum_op;
     return sum_op(saccum, predux(vaccum));
   }
 };

// --- Reducer: compare elements by condition
template<typename T>
struct scalar_comparer_op{
    scalar_comparer_op(std::function<T(T, T)> condition)
    :m_condition{condition}, is_first{true} {}

    void reduce(const T t, T* accum){
        if(is_first){
            *accum = t;
            is_first = false;
        }else{
            *accum = m_condition(t, *accum);
        }
    }
    template <typename Packet>
    void reducePacket(const Packet& packet, Packet* acc) const {
        (*acc) = Eigen::internal::padd<Packet>(*acc, packet);
    }

    T initialize() const{
        return T(0);
    }
    template <typename Packet>
    Packet initializePacket() const {
        float init = initialize();
        return Eigen::internal::pset1<Packet>(init);
    }

    T finalize(const T accum) const{
        return accum;
    }
    template <typename Packet>
    Packet finalizePacket(const Packet& acc) const {
        return acc;
    }
    template <typename Packet>
    T finalizeBoth(const T acc_val, const Packet& acc_packet) const {
        auto packet = Eigen::internal::predux(acc_packet);
        return this->custom_op(acc_val, packet);
    }
private:
    std::function<T(T, T)> m_condition;
    bool is_first;
};

// --- Reducer: cacluate min/max value rowwise or colwwise
template<typename T>
T min(const Tensor<T, 2>& t)
{
    array<Eigen::Index, 2> dims({0, 1});
    scalar_comparer_op<T> comparer([](T x, T y)->T {return x < y ? x: y;});
    Tensor<T, 0> temp = t.reduce(dims, comparer);
    return temp(0);
}

template<typename T>
T max(const Tensor<T, 2>& t)
{
    array<Eigen::Index, 2> dims({0, 1});
    scalar_comparer_op<T> comparer( [](T x, T y)->T {return x > y ? x: y;});
    Tensor<T, 0> temp = t.reduce(dims, comparer);
    std::cout << "temp " << temp << " " << temp(0) << "\n";
    return temp(0);
}
// ---

// --- Transpose tensor: by its first 2 dimensions
template<typename ArgType>
ArgType
transposed(const ArgType& t){
    Eigen::array<int, t.NumDimensions> shuffle_transpose;
    shuffle_transpose[0] = 1;
    shuffle_transpose[1] = 0;
    for(int i{2}; i < t.NumDimensions; i++)
        shuffle_transpose[i] = i;
    return t.shuffle(shuffle_transpose);
}

// --- Unary-Op: normalize tensor to range 0-1
struct max_normalize_op
{
    max_normalize_op(Tensor<float, 2> t)
        :m_max{max<float>(t)}, m_min{min<float>(t)}
    {
    }
    float operator()(float a) const{
        return (a- m_min) / (m_max-m_min);
    }
private:
    float m_max;
    float m_min;
};

// -- Slice tensor along given dimension:
template<int N>
Tensor<float, N> sliced(const Tensor<float, N>& arg, const std::vector<int>& indices, int dim){
  Eigen::array<Eigen::Index, N> out_size;
  for(int i{0}; i < N; i++){
    out_size[i] = arg.dimension(i);
  }
  out_size[dim] = indices.size();

  Tensor<float, N> out_slice(out_size);
  for(int i{0}; i < indices.size(); i++){
    out_slice.chip(i, dim) = arg.chip(indices[i], dim);
  }
  return out_slice;
}

template<int N>
Tensor<float, N> sliced(const Tensor<float, N>&& arg, const std::vector<int>& indices, int dim){
  Eigen::array<Eigen::Index, N> out_size;
  for(int i{0}; i < N; i++){
    out_size[i] = arg.dimension(i);
  }
  out_size[dim] = indices.size();

  Tensor<float, N> out_slice(out_size);
  for(int i{0}; i < indices.size(); i++){
    out_slice.chip(i, dim) = arg.chip(indices[i], dim);
  }
  return out_slice;
}

// -- Matrix product for tensors: l-value and r-value versions
#if 0
const Eigen::array<Eigen::IndexPair<int>, 1> product_dims = 
  {Eigen::IndexPair<int>(1, 0) };
Tensor<float, 2> prod1(const Tensor<float, 2>& a, const Tensor<float, 2>& b){
  return a.contract(b, product_dims);
}

Tensor<float, 2> prod1(Tensor<float, 2>&& a, Tensor<float, 2>&& b){
  return a.contract(b, product_dims);
}

Tensor<float, 2> prod1(Tensor<float, 2>&& a, const Tensor<float, 2>& b){
  return a.contract(b, product_dims);
}

Tensor<float, 2> prod1(const Tensor<float, 2>& a, Tensor<float, 2>&& b){
  return a.contract(b, product_dims);
}
#endif

#endif