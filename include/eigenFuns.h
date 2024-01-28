#ifndef FUNCTORS_H
#define FUNCTORS_H

#include <functional>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "typedefs.h"

// ---- Sum Matrix and Vector colwise or rowwise

template<typename ArgType1, typename ArgType2>
struct VecsumOp
{
 typedef typename ArgType1::Scalar _Scalar;
  VecsumOp(ArgType1& mat, ArgType2& vec, bool rowwise=true)
    :m_mat{mat}, m_vec{vec}, m_rows{mat.dimension(0)}, m_cols{mat.dimension(1)}, 
    m_rowwise{rowwise}
  {
    static_assert((ArgType1::NumDimensions == 2) && (ArgType2::NumDimensions == 1));
    if(rowwise){
      assert(m_mat.dimension(1)==m_vec.dimension(0));
    }else{
      assert(m_mat.dimension(0)==m_vec.dimension(0));
    }
  } 
  const _Scalar operator()(Eigen::Index idx) const{
    Eigen::Index row = idx % m_rows; 
    Eigen::Index col = idx / m_rows;
    return m_mat(row, col) + (m_rowwise?m_vec(col):m_vec(row));
  }
private:
  ArgType1 m_mat;
  ArgType2 m_vec;
  Eigen::Index m_rows;
  Eigen::Index m_cols;
  bool m_rowwise;
};

// L-value version
template<typename ArgType1, typename ArgType2>
auto vecSum(ArgType1& mat, ArgType2& vec, bool rowwise=true){
    return mat.nullaryExpr(VecsumOp<ArgType1, ArgType2>(mat, vec, rowwise));
}

// R-value version
template<typename ArgType1, typename ArgType2>
auto vecSum(ArgType1&& mat, ArgType2&& vec, bool rowwise=true){
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

// --- Reducer: cacluate min/max value
template<typename ArgType>
ArgType::Scalar min(const ArgType& t)
{
    //array<Eigen::Index, 2> dims({0, 1});
    std::array<Index, ArgType::NumDimensions> dims;
    for(size_t i{0}; i < ArgType::NumDimensions; i++){
      dims[i] = static_cast<Index>(i);
    }
    scalar_comparer_op<T> comparer([](T x, T y)->T {return x < y ? x: y;});
    Eigen::Tensor<T, 0> temp = t.reduce(dims, comparer);
    return temp(0);
}

template<typename ArgType>
ArgType::Scalar max(const ArgType& t)
{
    std::array<Index, ArgType::NumDimensions> dims;
    for(size_t i{0}; i < ArgType::NumDimensions; i++){
      dims[i] = static_cast<Index>(i);
    }
    scalar_comparer_op<T> comparer([](T x, T y)->T {return x > y ? x: y;});
    Eigen::Tensor<T, 0> temp = t.reduce(dims, comparer);
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
template<typename ArgType>
struct max_normalize_op
{
    max_normalize_op(ArgType t)
        :m_max{max<ArgType>(t)}, m_min{min<ArgType>(t)}
    {
    }
    float operator()(ArgType::Scalar a) const{
        return (a- m_min) / (m_max-m_min);
    }
private:
    ArgType::Scalar m_max;
    ArgType::Scalar m_min;
};

// -- Slice tensor along given dimension:
template<typename ArgType>
ArgType sliced(const ArgType& arg, const std::vector<int>& indices, int dim)
{
  assert(indices.size() < arg.dimension(dim));
  assert(*std::max_element(indices.begin(), indices.end()) < arg.dimension(dim));
  std::array<Eigen::Index, ArgType::NumDimensions> out_size;
  for(int i{0}; i < N; i++){
    out_size[i] = arg.dimension(i);
  }
  out_size[dim] = indices.size();

  ArgType out_slice(out_size);
  for(size_t i{0}; i < indices.size(); i++){
    out_slice.chip(i, dim) = arg.chip(indices[i], dim);
  }
  return out_slice;
}

template<typename ArgType>
ArgType sliced(const ArgType&& arg, const std::vector<int>& indices, int dim)
{
  assert(indices.size() < arg.dimension(dim));
  assert(*std::max_element(indices.begin(), indices.end()) < arg.dimension(dim));
  std::array<Eigen::Index, ArgType::NumDimensions> out_size;
  for(int i{0}; i < N; i++){
    out_size[i] = arg.dimension(i);
  }
  out_size[dim] = indices.size();

  ArgType out_slice(out_size);
  for(size_t i{0}; i < indices.size(); i++){
    out_slice.chip(i, dim) = arg.chip(indices[i], dim);
  }
  return out_slice;
}

#endif