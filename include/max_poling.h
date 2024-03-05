
#include <unsupported/Eigen/CXX11/Tensor>

namespace Eigen
{

template<typename ArgType1, typename ArgType2, typename ArgType3>
void max_pooling(ArgType1& input, Index ir, Index ic, Index depth, Index batch, 
	Index kr, Index kc, Index stride, ArgType2& output, ArgType3& argmax_out) {

    TensorRef<Tensor<float, internal::traits<ArgType2>::NumDimensions,
                    internal::traits<ArgType2>::Layout, Index>>
        output_ref(output);
	output.setConstant(Eigen::NumTraits<float>::lowest());

	Index outr = output_ref.dimension(1);
	Index outc = output_ref.dimension(2);
	
	for (Index i{ 0 }; i < batch; i++) {
		for (Index k{ 0 }; k < depth; k++) {
			for (Index r{ 0 }; r < ir; r++) {
				for (Index c{ 0 }; c < ic; c++) {
					// This is the windows of the output affected by the input at
					// r, c
					Index hstart = (r - kr) < 0 ? 0 : (r - kr) / stride + 1;
					//Index hend = (r + kr) > outr ? outr : r / stride + 1;
					Index hend = std::min(outr, r / stride + 1);
					Index wstart = (c - kc) < 0 ? 0 : (c - kc) / stride + 1;
					//Index wend = (c + kc) > outc ? outc : c / stride + 1;
					Index wend = std::min(outc, c / stride + 1);
					//std::cout << r << ", " << c << ", " << k << ", " << i << "\n";
					for (Index h{ hstart }; h < hend; h++) {
						for (Index w{ wstart }; w < wend; w++) {
							Index idx_flat = r + c * ir + k * ic * ir + i * ic * ir * depth;
							if (input(idx_flat) > output(0, h, w, k, i)) {
								output(0, h, w, k, i) = input(idx_flat);
								argmax_out(0, h, w, k, i) = idx_flat;
							}
						}
					}
				}
			}
		}
	}
}
}
