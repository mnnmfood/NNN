#ifndef CONVOLV_H
#define CONVOLV_H

#include <unsupported/Eigen/CXX11/Tensor>

namespace Eigen{

template<typename ArgType, typename KerType>
EIGEN_ALWAYS_INLINE static const
TensorReshapingOp<
    const DSizes<Index, internal::traits<ArgType>::NumDimensions>,
    const TensorForcedEvalOp<
        const TensorShufflingOp<
            const DSizes<Index, internal::traits<ArgType>::NumDimensions+1>,
            const TensorReshapingOp<
                const DSizes<Index,internal::traits<ArgType>::NumDimensions+1>,
                const TensorContractionOp<
                    const array<IndexPair<int>,1>,
                    const TensorReshapingOp<const DSizes<Index,2>,
                        KerType>,
                    const TensorReshapingOp<const DSizes<Index,2>,
                        const TensorImagePatchOp<-1,-1,
                            const ArgType>>>>>>>
convolveBatch(const ArgType& input, KerType& kernels){
    typedef typename internal::traits<ArgType>::Index TensorIndex;
    typedef typename internal::traits<ArgType>::Scalar OutScalar;

    TensorRef<Tensor<typename internal::traits<ArgType>::Scalar,
                    internal::traits<ArgType>::NumDimensions,
                    internal::traits<ArgType>::Layout, TensorIndex>>
        input_ref(input);
    TensorRef<Tensor<OutScalar, internal::traits<KerType>::NumDimensions,
                    internal::traits<KerType>::Layout, TensorIndex>>
        kernels_ref(kernels);

    assert(internal::traits<KerType>::NumDimensions == 4);
    const TensorIndex num_dims = internal::traits<ArgType>::NumDimensions; 
    const array<IndexPair<int>, 1> contract_dims{
        IndexPair(1, 0)
    };

    const TensorIndex depth = kernels_ref.dimension(0);
    const TensorIndex kr = kernels_ref.dimension(2);
    const TensorIndex kc = kernels_ref.dimension(3);
    const TensorIndex in_depth = input_ref.dimension(3);
    const TensorIndex channels = input_ref.dimension(0);
    const TensorIndex batch = input_ref.dimension(4);

    DSizes<TensorIndex, internal::traits<ArgType>::NumDimensions> out_shape;
    out_shape[0] = 1;
    out_shape[1] = input_ref.dimension(1) - kr + 1;
    out_shape[2] = input_ref.dimension(2) - kc + 1;
    out_shape[3] = depth * in_depth;
    out_shape[4] = batch;

    DSizes<TensorIndex, 2> input_contract_shape;
    input_contract_shape[0] = channels * kr * kc;
    input_contract_shape[1] = out_shape[1] * out_shape[2] * in_depth * batch;

    DSizes<TensorIndex, 2> kernels_contract_shape;
    kernels_contract_shape[0] = depth;
    kernels_contract_shape[1] = channels * kr * kc;
    
    DSizes<TensorIndex, 6> pos_contract_shape;
    pos_contract_shape[0] = depth;
    pos_contract_shape[1] = channels;
    pos_contract_shape[2] = input_ref.dimension(1) - kr + 1;
    pos_contract_shape[3] = input_ref.dimension(2) - kc + 1;
    pos_contract_shape[4] = in_depth;
    pos_contract_shape[5] = batch;
    
    // [depth, channels, ir, ic, in_depth, batch] to
    // [channels, ir, ic, depth, in_depth, batch] 
    const DSizes<TensorIndex, 6> pos_contract_shuffle{1, 2, 3, 0, 4, 5};

    const TensorIndex padr = 0, padc = 0;

    return  kernels.reshape(kernels_contract_shape)
        .contract(input
            .extract_image_patches(kr, kc, 1, 1,
                1, 1, 1, 1,
                padr, padr, padc, padc,
                OutScalar(0))
            .reshape(input_contract_shape), 
        contract_dims)
        .reshape(pos_contract_shape)
        .shuffle(pos_contract_shuffle).eval()
        .reshape(out_shape);
}

template<typename ArgType1, typename ArgType2>
inline static const
TensorReshapingOp<const DSizes<Index, internal::traits<ArgType1>::NumDimensions - 1>,
    const TensorContractionOp<const array<IndexPair<Index>, 1>,
		const TensorReshapingOp<const DSizes<Index, 2>,
			const TensorImagePatchOp<-1, -1,
			const TensorReshapingOp<
				const DSizes<Index, internal::traits<ArgType2>::NumDimensions - 1>,
				const TensorReverseOp<
					const DSizes<bool, internal::traits<ArgType2>::NumDimensions>,
					const ArgType2>>>>,
    const TensorReshapingOp<const DSizes<Index, 2>,
		const TensorForcedEvalOp<
			const TensorShufflingOp<
				const DSizes<Index, internal::traits<ArgType1>::NumDimensions>,
				const ArgType1>>>>>
backwardsConvolveInput(const ArgType1& grad, const ArgType2& kernels,
    Index& input_r, Index& input_c){
    typedef typename internal::traits<ArgType1>::Index TensorIndex;
    typedef typename internal::traits<ArgType2>::Scalar OutScalar;
    
    TensorRef<Tensor<typename internal::traits<ArgType1>::Scalar,
        internal::traits<ArgType1>::NumDimensions,
        internal::traits<ArgType1>::Layout, TensorIndex>>
        grad_ref(grad);
    TensorRef<Tensor<OutScalar, internal::traits<ArgType2>::NumDimensions,
        internal::traits<ArgType2>::Layout, TensorIndex>>
        kernels_ref(kernels);

    assert(grad_ref.dimension(3) == kernels_ref.dimension(0));
    const TensorIndex channels = grad_ref.dimension(0);
    const TensorIndex gradr = grad_ref.dimension(1);
    const TensorIndex gradc = grad_ref.dimension(2);
    const TensorIndex batch = grad_ref.dimension(4);
    const TensorIndex kr = kernels_ref.dimension(2);
    const TensorIndex kc = kernels_ref.dimension(3);
    const TensorIndex depth = kernels_ref.dimension(0);
    const TensorIndex patches = input_r * input_c;

    const DSizes<bool, 4> kern_reverse{ false, false, true, true };
    // [depth, channels, kr, kc] to
    // [depth*channels, kr, kc]
    DSizes<TensorIndex, 3> kern_shape;
    kern_shape[0] = depth * channels;
    kern_shape[1] = kr;
    kern_shape[2] = kc;
    DSizes<TensorIndex, 2> kern_contract_shape;
    kern_contract_shape[0] = depth * channels * gradr * gradc;
    kern_contract_shape[1] = patches;
    
    // [channels, or, oc, depth, batch] to
    // [depth, channels, or, oc, batch] to
    const DSizes<TensorIndex, 5> grad_shuffle{ 3, 0, 1, 2, 4 };
    DSizes<TensorIndex, 2> grad_contract_shape;
    grad_contract_shape[0] = depth * channels * gradr * gradc;
    grad_contract_shape[1] = batch;
    
    // Full convolution
    const TensorIndex padr = gradr - 1;
    const TensorIndex padc = gradc - 1;

    const array<IndexPair<Index>, 1> contract_dims{
        IndexPair<Index>(0, 0)
    };

    DSizes<TensorIndex, internal::traits<ArgType1>::NumDimensions-1> out_shape;
    out_shape[0] = channels;
    out_shape[1] = input_r;
    out_shape[2] = input_c;
    out_shape[3] = batch;

    auto kernels_reversed = kernels.reverse(kern_reverse)
        .reshape(kern_shape);
    auto grad_reshaped = grad
        .shuffle(grad_shuffle).eval()
        .reshape(grad_contract_shape);

    return kernels_reversed
            .extract_image_patches(gradr, gradc, 1, 1,
                1, 1, 1, 1,
                padr, padr, padc, padc,
                OutScalar(0))
            .reshape(kern_contract_shape)
        .contract(grad_reshaped,
            contract_dims)
        .reshape(out_shape);
}

template<typename ArgType1, typename ArgType2>
inline static const
TensorReshapingOp<const DSizes<Index, 4>,
    const TensorContractionOp<const array<IndexPair<Index>, 1>,
		const TensorReshapingOp<const DSizes<Index, 2>,
			const TensorForcedEvalOp<
				const TensorShufflingOp<const array<Index,
    internal::traits<ArgType2>::NumDimensions>,
				const ArgType2>>>,
		const TensorReshapingOp<const DSizes<Index, 2>,
			const TensorImagePatchOp<Dynamic, Dynamic,
				const TensorForcedEvalOp<
				const TensorShufflingOp<
					const array<Index, 
    internal::traits<ArgType1>::NumDimensions>,
					const ArgType1>>>>>>
backwardsConvolveKernel(const ArgType1& input, const ArgType2& output, 
                        Index kr, Index kc){
    typedef typename internal::traits<ArgType1>::Index TensorIndex;
    typedef typename internal::traits<ArgType2>::Scalar OutScalar;

    TensorRef<Tensor<typename internal::traits<ArgType1>::Scalar,
                    internal::traits<ArgType1>::NumDimensions,
                    internal::traits<ArgType1>::Layout, TensorIndex>>
        input_ref(input);
    TensorRef<Tensor<OutScalar, internal::traits<ArgType2>::NumDimensions,
                    internal::traits<ArgType2>::Layout, TensorIndex>>
        output_ref(output);


    const TensorIndex outr = output_ref.dimension(1);
    const TensorIndex outc = output_ref.dimension(2);
    const TensorIndex depth = output_ref.dimension(3);
    const TensorIndex channels = input_ref.dimension(0);
    const TensorIndex inr = input_ref.dimension(1);
    const TensorIndex inc = input_ref.dimension(2);
    const TensorIndex batch = input_ref.dimension(3);
    const TensorIndex patches = kr * kc;

    // [channels, input_rows, input_cols, batch] to
    // [batch, nput_rows, input_cols, channels]
    array<TensorIndex, 4> input_shuffle ({3, 1, 2, 0});

    DSizes<TensorIndex, 2> input_contract_shape;
    input_contract_shape[0] = channels * batch * outr * outc;
    input_contract_shape[1] = patches;

    // [ker_depth, output_rows, output_cols, batch] to 
    // [ker_depth, batch, output_rows, output_cols]

    // [channels, output_rows, output_cols, ker_depth, batch] to 
    // [channels, batch, output_rows, ker_depth, output_cols]
    array<TensorIndex, 5> output_shuffle ({3, 4, 0, 1, 2});
    DSizes<TensorIndex, 2> output_contract_shape;
    output_contract_shape[0] = depth;
    output_contract_shape[1] = channels * outr * outc * batch;

    DSizes<TensorIndex, 4> ker_shape;
    ker_shape[0] = depth;
    ker_shape[1] = channels;
    ker_shape[2] = kr;
    ker_shape[3] = kc;

    const TensorIndex padr{ std::max<TensorIndex>(outr - inr, 0) };
    const TensorIndex padc{ std::max<TensorIndex>(outc - inc, 0) };

    const array<IndexPair<Index>, 1> contract_dims {
        IndexPair<Index>(1, 0)
    };

    const auto input_shuffled = input
        .shuffle(input_shuffle).eval();
    const auto output_shuffled = output
        .shuffle(output_shuffle).eval()
        .reshape(output_contract_shape);
    return output_shuffled
        .contract(input_shuffled
            .extract_image_patches(outr, outc, 1, 1,
                1, 1, 1, 1,
                padr, padr, padc, padc,
                OutScalar(0))
            .reshape(input_contract_shape),
            contract_dims)
        .reshape(ker_shape);
}
}

#endif