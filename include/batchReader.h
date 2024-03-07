#ifndef BATCH_READER
#define BATCH_READER

#include <filesystem>
#include <random>
#include <exception>
#include "typedefs.h"
#include "utils.h"

namespace fs = std::filesystem;

class BatchPNGReader
{
	std::mt19937 _gen;
	std::string _ext = ".png";
	std::string _parent_dir;
	std::vector<std::string> _path_arr;

	Index _batch;
	size_t _total_size;
	size_t _index{ 0 };

public: 
	BatchPNGReader(std::string& dir, Index batch)
		:_parent_dir{dir}, _batch{batch}
	{
		std::random_device rd{};
		_gen = std::mt19937{rd()};
		
		for (const auto& entry : fs::directory_iterator(_parent_dir)){
			if ((entry.path().extension().compare(_ext)) == 0) {
				_path_arr.push_back((entry.path().string()));
			}
		}
		_total_size = _path_arr.size();
		if (_total_size < batch) {
			throw(std::runtime_error("No images found"));
		}
		// shuffle before loading any images
        std::shuffle(_path_arr.begin(), _path_arr.end(), _gen);
	}

	int operator++() { 
		if ((_index + _batch) > _total_size) {
			return 0;
		}
		_index += _batch; 
		return 1;
	}
	int operator++(int) { 
		return ++(*this);
	} 

	auto get() {
		return imread_bulk(_path_arr.begin() + _index, _path_arr.begin() + _index + _batch);
	}

	void reset() {
		_index = 0;
        std::shuffle(_path_arr.begin(), _path_arr.end(), _gen);
	}
};

#endif
