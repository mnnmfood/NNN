#ifndef BATCH_READER
#define BATCH_READER

#include <filesystem>
#include <random>
#include <exception>
#include "typedefs.h"
#include "utils.h"

namespace fs = std::filesystem;

template<typename ptr_t = std::pair<int, std::string>>
void imread_bulk(ptr_t* begin, ptr_t* end, Tensor<byte, 3>& images){
	try {
		// create reader and check image shape
		std::ifstream fp{ begin->second, std::ios::binary };
		png::PNGReader reader{ fp };
		const Eigen::Index batch = static_cast<Index>(end - begin);
		const Eigen::Index width = reader.m_info.width;
		const Eigen::Index height = reader.m_info.height;
		const Eigen::Index total_bytes = width * height * sizeof(byte);
		const Eigen::array<Eigen::Index, 3> i_shape({
			height,
			width,
			batch
			});
		// resize tensor and get raw array
		images = Tensor<byte, 3>(i_shape);
		byte* im_arr = images.data();

		int i{ 0 };
		for (; begin != end; begin++, i++) {
			std::ifstream fpi(begin->second, std::ios::binary);
			reader.reset(fpi);
			reader.read_arr(im_arr + total_bytes * i, width, height, PNG_COLOR_TYPE_GRAY);
		}
	}
	catch (std::exception& e) {
			std::cout << "Error reading image " + begin->second << "\n";
			throw(std::runtime_error("Error Reading Images"));
	}
}

struct BatchPNGIterator
{

	typedef Tensor<byte, 3> out_data_t;
	typedef Tensor<float, 2> out_label_t;
	typedef std::pair<int, std::string> it_t;

	BatchPNGIterator(it_t* begin, Index batch, int num_labels)
		:_begin{ begin }, _batch{ batch }, _num_labels{ num_labels },
		_labels(static_cast<Index>(num_labels), batch)
	{
	}

	BatchPNGIterator& operator ++() {
		_begin += _batch;
		return *this;
	}

	BatchPNGIterator operator ++(int) {
		BatchPNGIterator temp = *this;
		++(*this);
		return temp;
	}

	BatchPNGIterator& operator --() {
		_begin -= _batch;
		return *this;
	}

	BatchPNGIterator operator --(int) {
		BatchPNGIterator temp = *this;
		--(*this);
		return temp;
	}
	
	friend bool operator==(BatchPNGIterator& a, BatchPNGIterator& b) { return a._begin == b._begin; }
	friend bool operator!=(BatchPNGIterator& a, BatchPNGIterator& b) { return a._begin != b._begin; }
	
	const Tensor<float, 2>& labels() {
		//one-hot encode labels
		_labels.setConstant(0.0f);
		for (int i{ 0 }; i < _batch; ++i) {
			it_t* it = _begin + i;
			_labels(it->first, i) = 1;
		}
		return _labels;
	}

	const Tensor<byte, 3>& images() {
		imread_bulk(_begin, _begin + _batch, _images);
		return _images;
	}

private:
	int _test{ 0 };
	it_t* _begin;
	Index _batch;
	int _num_labels;
	Tensor<byte, 3> _images;
	Tensor<float, 2> _labels;
};

class BatchPNGReader
{
	typedef BatchPNGIterator it;
	typedef std::pair<int, std::string> data_t;
	std::mt19937 _gen;
	std::string _ext = ".png";
	std::string _parent_dir;
	std::vector<data_t> _path_arr;

	data_t* _data;
	Index _batch;
	Eigen::Index _total_size;
	int _labels{ 0 };

public:
	typedef it::out_data_t out_data_t;
	typedef it::out_label_t out_label_t;

	BatchPNGReader(std::string& dir, Index batch)
		:_parent_dir{ dir }, _batch{ batch }
	{
		std::random_device rd{};
		_gen = std::mt19937{ rd() };

		for (const auto& p_entry : fs::directory_iterator(_parent_dir)) {
			if (fs::is_directory(p_entry)) {
				for (const auto& c_entry : fs::directory_iterator(p_entry)) {
					if ((c_entry.path().extension().compare(_ext)) == 0) {
						_path_arr.push_back(
							std::make_pair(_labels, c_entry.path().string())
						);
					}
				}
				_labels++;
			}
		}

		_total_size = _path_arr.size();
		_data = _path_arr.data();
		if (_total_size < batch) {
			throw(std::runtime_error("No images found"));
		}
		// shuffle before loading any images
		std::shuffle(_path_arr.begin(), _path_arr.end(), _gen);
	}

	BatchPNGIterator begin() { return BatchPNGIterator(_data, _batch, _labels); }
	BatchPNGIterator end() {
		// address used to stop iteration
		int misal = _total_size % _batch;
		int last_idx = misal == 0 ? _total_size : _total_size - _batch + misal;
		return BatchPNGIterator(_data + last_idx, _batch, _labels);
	}

	void reset() {
		std::shuffle(_path_arr.begin(), _path_arr.end(), _gen);
	}

	Eigen::Index batch() {
		return _batch;
	}

	Eigen::Index size() {
		return _total_size;
	}
};

#endif
