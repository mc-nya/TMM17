#pragma once
#include "caffe/common.hpp"
#include "caffe/layers/input_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/dropout_layer.hpp"

// add layer names for caffe(windows version)
namespace caffe
{
	extern INSTANTIATE_CLASS(InputLayer);
	extern INSTANTIATE_CLASS(InnerProductLayer); 
	extern INSTANTIATE_CLASS(DropoutLayer);
	//REGISTER_LAYER_CLASS(Input);
	//extern INSTANTIATE_CLASS(PoolingLayer);

}