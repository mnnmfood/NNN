#ifndef BATCH_READER
#define BATCH_READER

#include <filesystem>
#include <random>
#include <exception>
#include "layer_traits.h"
#include "typedefs.h"
#include "utils.h"

template<class Derived>
class BatchReader
{
public:
	typedef typename traits<Derived>::iterator it;
	typedef typename traits<Derived>::data_t data_t;

	std::mt19937 _gen;
	std::vector<data_t> _path_arr;

	data_t* _data;
	Index _batch;
	Index _total_size;

public:
	typedef typename traits<Derived>::out_data_t out_data_t;
	typedef typename traits<Derived>::out_label_t out_label_t;

	BatchReader(Index batch): _batch{ batch }{}

	it begin() {
		return static_cast<Derived*>(this)->iter(_data, _batch);
	}
	it end() {
		// address used to stop iteration
		int misal = _total_size % _batch;
		int last_idx = misal == 0 ? _total_size : _total_size - _batch + misal;
		return static_cast<Derived*>(this)->iter(_data + last_idx, _batch);
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
