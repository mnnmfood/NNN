#ifndef TEST_OPS_H
#define TEST_OPS_H

#include <string>
#include <string_view>
#include <cmath>
#include "typedefs.h"
#include "convolutions.h"
#include "layer_activations.h"
#include "cost_funs.h"


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

	Index im_size_ind = static_cast<Index>(im_size);
	input.device(*device) = Eigen::backwardsConvolveInput(output, kernel, im_size_ind, im_size_ind);

	for (int b{ 0 }; b < batch; b++) {
		for (int ir {0}; ir < im_size; ir ++) {
			for (int ic {0}; ic < im_size; ic ++) {
				float expected = 0.0f;
				for (int kr{ 0 }; kr < ker_size; kr++) {
					for (int kc{ 0 }; kc < ker_size; kc++) {
						int outr = ir - kr;
						int outc = ic - kc;
						if (outr >= 0 && outr < out_size &&
							outc >= 0 && outc < out_size) {
							for (int dk{ 0 }; dk < ker_depth; dk++) {
								expected +=
									output(0, outr, outc, dk, b) * kernel(dk, 0, kr, kc);
							}
						}
					}
				}
				AssertAprox(input(0, ir, ic, b), expected, "backwards input");
			}
		}
	}
}

void testBackwardsKernel(int im_size, int batch, int ker_size, int ker_depth, ThreadPoolDevice* device) {
	int out_size = im_size - ker_size + 1;
	int out_depth = ker_depth;
	Tensor<float, 4> input(1, im_size, im_size, batch);
	input.setRandom();
	Tensor<float, 4> kernel(ker_depth, 1, ker_size, ker_size);
	kernel.setConstant(0.0f);
	Tensor<float, 5> output(1, out_size, out_size, out_depth, batch);
	output.setRandom();

	Index ker_size_ind = static_cast<Index>(ker_size);
	kernel.device(*device) = Eigen::backwardsConvolveKernel(input, output, ker_size_ind, ker_size_ind);

	for (int d{ 0 }; d < ker_depth; d++) {
		for (int kr {0}; kr < ker_size; kr ++) {
			for (int kc {0}; kc < ker_size; kc ++) {
				float expected = 0.0f;
				int startr = kr;
				int endr = kr + out_size;
				int startc = kc;
				int endc = kc + out_size;
				for (int ir{ startr }, outr{ 0 }; ir < endr; ir++, outr++) {
					for (int ic{ startc }, outc{ 0 }; ic < endc; ic++, outc++) {
						for (int b{ 0 }; b < batch; b++) {
							expected +=
								input(0, ir, ic, b) * output(0, outr, outc, d, b);
						}
					}
				}
				AssertAprox(kernel(d, 0, kr, kc), expected, "convolution");
			}
		}
	}
}

void testSoftMax(int size, int batch, ThreadPoolDevice* device) {
	Tensor<float, 2> input(size, batch);
	input.setRandom();
	Tensor<float, 2> output(size, batch);
	softmax_fun(input, output, device);
	for (int b{ 0 }; b < batch; b++) {
		float sum = 0;
		for (int i{ 0 }; i < size; i++) {
			sum += std::exp(input(i, b));
		}
		for (int i{ 0 }; i < size; i++) {
			float expected = std::exp(input(i, b)) / sum;
			AssertAprox(output(i, b), expected, "softmax");
		}
	}

	softmax_grad_fun(input, output, device);

	for (int b{ 0 }; b < batch; b++) {
		float sum = 0;
		for (int i{ 0 }; i < size; i++) {
			sum += std::exp(input(i, b));
		}
		for (int i{ 0 }; i < size; i++) {
			float softmax = std::exp(input(i, b)) / sum;
			float expected = softmax - softmax * softmax;
			AssertAprox(output(i, b), expected, "softmax");
		}
	}

}

void testSigmoid(int size, int batch, ThreadPoolDevice* device) {
	Tensor<float, 2> input(size, batch);
	input.setRandom();
	Tensor<float, 2> output(size, batch);
	sigmoid_fun(input, output, device);
	for (int b{ 0 }; b < batch; b++) {
		for (int i{ 0 }; i < size; i++) {
			float expected = 1.0f / (1.0f + std::exp(-input(i, b)));
			AssertAprox(output(i, b), expected, "sigmoid");
		}
	}

	sigmoid_grad_fun(input, output, device);

	for (int b{ 0 }; b < batch; b++) {
		for (int i{ 0 }; i < size; i++) {
			float expected = 1.0f / (1.0f + std::exp(-input(i, b)));
			expected = expected * (1 - expected);
			AssertAprox(output(i, b), expected, "sigmoid grad");
		}
	}

}

void testTanh(int size, int batch, ThreadPoolDevice* device) {
	Tensor<float, 2> input(size, batch);
	input.setRandom();
	Tensor<float, 2> output(size, batch);
	tanh_fun(input, output, device);
	for (int b{ 0 }; b < batch; b++) {
		for (int i{ 0 }; i < size; i++) {
			float expected = std::tanh(input(i, b));
			AssertAprox(output(i, b), expected, "tanh");
		}
	}

	tanh_grad_fun(input, output, device);

	for (int b{ 0 }; b < batch; b++) {
		for (int i{ 0 }; i < size; i++) {
			float expected = std::tanh(input(i, b));
			expected = 1 - expected * expected;
			AssertAprox(output(i, b), expected, "tanh grad");
		}
	}
}

void testReLu(int size, int batch, ThreadPoolDevice* device) {
	Tensor<float, 2> input(size, batch);
	input.setRandom();
	input = input - 0.5f;
	Tensor<float, 2> output(size, batch);
	relu_fun(input, output, device);
	for (int b{ 0 }; b < batch; b++) {
		for (int i{ 0 }; i < size; i++) {
			float expected = std::max(input(i, b), 0.0f);
			AssertAprox(output(i, b), expected, "tanh");
		}
	}
	relu_grad_fun(input, output, device);
	for (int b{ 0 }; b < batch; b++) {
		for (int i{ 0 }; i < size; i++) {
			float expected = input(i, b) > 0 ? 1: 0;
			AssertAprox(output(i, b), expected, "tanh grad");
		}
	}

}

void testMSE(int size, int batch, ThreadPoolDevice* device) {
	Tensor<float, 2> input(size, batch);
	input.setRandom();
	Tensor<float, 2> output(size, batch);
	output.setRandom();
	Tensor<float, 0> cost;
	cost.device(*device) = mse_fun(input, output, device);
	float expected = 0;
	for (int b{ 0 }; b < batch; b++) {
		float accum = 0;
		for (int i{ 0 }; i < size; i++) {
			float x = input(i, b);
			float y = output(i, b);
			accum += (y - x) * (y - x);
		}
		expected += std::sqrt(accum);
	}
	AssertAprox(cost(0), expected, "MSE");
	
	Tensor<float, 2> grad(size, batch);
	mse_grad_fun(input, output, grad, device);

	for (int b{ 0 }; b < batch; b++) {
		for (int i{ 0 }; i < size; i++) {
			float x = input(i, b);
			float y = output(i, b);
			expected = x - y;
			AssertAprox(grad(i, b), expected, "MSE grad");
		}
	}

}

void testCrossEntropy(int size, int batch, ThreadPoolDevice* device) {
	Tensor<float, 2> input(size, batch);
	input.setRandom();
	Tensor<float, 2> output(size, batch);
	output.setRandom();
	Tensor<float, 0> cost;
	cost.device(*device) = cross_entropy_fun(input, output, device);
	float expected = 0;
	for (int b{ 0 }; b < batch; b++) {
		float sum = 0;
		for (int i{ 0 }; i < size; i++) {
			sum += std::exp(input(i, b));
		}
		for (int i{ 0 }; i < size; i++) {
			//float x = input(i, b);
			float y = output(i, b);
			float x = std::exp(input(i, b)) / sum;
			expected += -y * std::log(x) + (y - 1) * std::log(1 - x);
		}
	}
	AssertAprox(cost(0), expected, "MSE");

	Tensor<float, 2> grad1(size, batch);
	cross_entropy_grad_fun(input, output, grad1, device, true);
	Tensor<float, 2> grad2(size, batch);
	cross_entropy_grad_fun(input, output, grad2, device, false);
	for (int b{ 0 }; b < batch; b++) {
		for (int i{ 0 }; i < size; i++) {
			float x = input(i, b);
			float y = output(i, b);
			expected = x - y;
			AssertAprox(grad1(i, b), expected, "MSE grad");
			expected = (1 - y) / (1 - x) - y / x;
			AssertAprox(grad2(i, b), expected, "MSE grad");
		}
	}

}


void testAllOps() {

        const int pool_n{ 8 };
        const int thread_n{ 4 };
        ThreadPool pool(pool_n);
        ThreadPoolDevice device(&pool, thread_n);
		int im_size{ 10 }, im_depth{ 3 }, batch{ 10 };
		int ker_size{ 10 }, ker_depth{ 3 };
		testConvoution(im_size, im_depth, batch, ker_size, ker_depth, &device);
		testBackwardsInput(im_size, batch, ker_size, ker_depth, &device);
		testBackwardsKernel(im_size, batch, ker_size, ker_depth, &device);
		
		int size{ 5 };
		batch = 10;
		testSoftMax(size, batch, &device);
		testSigmoid(size, batch, &device);
		testTanh(size, batch, &device);
		testReLu(size, batch, &device);

		testMSE(size, batch, &device);
		testCrossEntropy(size, batch, &device);
		std::cout << "Sucess\n\n";
}

#endif
