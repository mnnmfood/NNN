#ifndef TEST_OPS_H
#define TEST_OPS_H

#include <string>
#include <string_view>
#include "typedefs.h"
#include "convolutions.h"


static constexpr float TestPrecision = 1e-3;

#define ASSERT_WITH_MSG(cond, msg) do \
{ if (!(cond)) { std::ostringstream str; str << msg; std::cerr << str.str(); std::abort(); } \
} while(0)

void AssertAprox(float a, float b, std::string&& test_name) {
	ASSERT_WITH_MSG(std::abs(a - b) < TestPrecision, "Test " + test_name + " failed");
}

void testConvoution(int im_size, int depth, int batch, int ker_size, int ker_depth, ThreadPoolDevice* device) {
	int out_size = im_size - ker_size + 1;
	int out_depth = depth * ker_depth;
	Tensor<float, 5> input(1, im_size, im_size, depth, batch);
	input.setRandom();
	Tensor<float, 4> kernel(ker_depth, 1, ker_size, ker_size);
	kernel.setRandom();
	Tensor<float, 5> output(1, out_size, out_size, out_depth, batch);

	//Tensor<float, 5> expected(1, out_size, out_size, out_depth, batch);
	//expected.setConstant(0.0f);
	output.device(*device) = Eigen::convolveBatch(input, kernel);

	for (int b{ 0 }; b < batch; b++) {
		for (int d{ 0 }; d < depth; d++) {
			for (int dk{ 0 }; dk < ker_depth; dk++) {
				int od = d * ker_depth + dk;
				for (int oh {0}; oh < out_size; oh ++) {
					for (int oc {0}; oc < out_size; oc ++) {
						float expected = 0.0f;
						int startr = oh;
						int endr = startr + ker_size;
						int startc = oc;
						int endc = startc + ker_size;
						for (int ir{ startr }, kr{ 0 }; ir < endr; ir++, kr++) {
							for (int ic{ startc }, kc{ 0 }; ic < endc; ic++, kc++) {
								expected +=
									input(0, ir, ic, d, b) * kernel(dk, 0, kr, kc);
							}
						}
						AssertAprox(output(0, oh, oc, od, b), expected, "convolution");
					}
				}
			}
		}
	}
}


void testBackwardsInput(int im_size, int batch, int ker_size, int ker_depth, ThreadPoolDevice* device) {
	int out_size = im_size - ker_size + 1;
	int out_depth = ker_depth;
	Tensor<float, 4> input(1, im_size, im_size, batch);
	Tensor<float, 4> kernel(ker_depth, 1, ker_size, ker_size);
	kernel.setRandom();
	Tensor<float, 5> output(1, out_size, out_size, out_depth, batch);
	output.setRandom();

	//Tensor<float, 5> expected(1, out_size, out_size, out_depth, batch);
	//expected.setConstant(0.0f);
	Index im_size_ind = static_cast<Index>(im_size);
	input.device(*device) = Eigen::backwardsConvolveInput(output, kernel, im_size_ind, im_size_ind);

	for (int b{ 0 }; b < batch; b++) {
		for (int dk{ 0 }; dk < ker_depth; dk++) {
			for (int ir {0}; ir < im_size; ir ++) {
				for (int ic {0}; ic < im_size; ic ++) {
					float expected = 0.0f;
					for (int kr{ 0 }; kr < ker_size; kr++) {
						for (int kc{ 0 }; kc < ker_size; kc++) {
							//int rkr = std::abs(kr - ker_size);
							//int rkc = std::abs(kc - ker_size);
							int outr = ir - kr;
							int outc = ic - kc;
							if (outr >= 0 && outr < out_size &&
								outc >= 0 && outc < out_size) {
								std::cout << kr << ", " << kc << ";";
								std::cout << outr << ", " << outc << "\n";
								expected +=
									output(0, outr, outc, dk, b) * kernel(dk, 0, kr, kc);
							}
						}
					}
					//AssertAprox(input(0, ir, ic, b), expected, "convolution");
				}
			}
		}
	}
}

void testAllOps() {

        const int pool_n{ 8 };
        const int thread_n{ 4 };
        ThreadPool pool(pool_n);
        ThreadPoolDevice device(&pool, thread_n);
		//testConvoution(10, 1, 1, 3, 1, &device);
		testBackwardsInput(3, 1, 2, 1, &device);
}

#endif
