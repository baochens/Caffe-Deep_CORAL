#ifndef CAFFE_CORAL_LOSS_LAYER_HPP_
#define CAFFE_CORAL_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class CORALLossLayer : public LossLayer<Dtype> {
 public:
  explicit CORALLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CORALLoss"; }
  /**
   * Similar to EuclideanLossLayer, in CORALLoss we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc CORALLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  Blob<Dtype> cov_s;
  Blob<Dtype> cov_t;
  Blob<Dtype> mean_s;
  Blob<Dtype> mean_t;
  Blob<Dtype> diff_data_s;
  Blob<Dtype> diff_data_t;
  Blob<Dtype> square_mean_s;
  Blob<Dtype> square_mean_t;
  Blob<Dtype> bp_mean_s;
  Blob<Dtype> bp_mean_t;
  Blob<Dtype> bp_der_s;
  Blob<Dtype> bp_der_t;
  Blob<Dtype> identity;
};

}  // namespace caffe

#endif  // CAFFE_CORAL_LOSS_LAYER_HPP_
