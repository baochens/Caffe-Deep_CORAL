#include <vector>

#include "caffe/layers/coral_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <algorithm>

namespace caffe {

template <typename Dtype>
void CORALLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  // CHECK_EQ might not be necessary as the size of the covariance matrices of
  //source and target are always the same even if the batch size is different.
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int dim = count / num;
  diff_.Reshape(dim, dim, 1, 1);
  cov_s.Reshape(dim, dim, 1, 1);
  cov_t.Reshape(dim, dim, 1, 1);
  mean_s.Reshape(dim, 1, 1, 1);
  mean_t.Reshape(dim, 1, 1, 1);
  square_mean_s.Reshape(dim, dim, 1, 1);
  square_mean_t.Reshape(dim, dim, 1, 1);
  diff_data_s.Reshape(num, dim, 1, 1);
  diff_data_t.Reshape(num, dim, 1, 1);
  identity.Reshape(num, 1, 1, 1);
  bp_mean_s.Reshape(dim, num, 1, 1);
  bp_mean_t.Reshape(dim, num, 1, 1);
  bp_der_s.Reshape(dim, num, 1, 1);
  bp_der_t.Reshape(dim, num, 1, 1);
}

template <typename Dtype>
void CORALLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int dim = count / num;
  const int size_cov = dim * dim;
  // calculating D'D for source and target
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, dim, dim, num, 1., bottom[0]->cpu_data(), bottom[0]->cpu_data(), 0., cov_s.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, dim, dim, num, 1., bottom[1]->cpu_data(), bottom[1]->cpu_data(), 0., cov_t.mutable_cpu_data());
  // divide D'D by (num-1)
  caffe_scal<Dtype>(size_cov, Dtype(1./(num-1)), cov_s.mutable_cpu_data());
  caffe_scal<Dtype>(size_cov, Dtype(1./(num-1)), cov_t.mutable_cpu_data());
  // identity is a row vector of 1s
  caffe_set(num, Dtype(1.), identity.mutable_cpu_data());
  // calculate the mean of D per column
  caffe_cpu_gemv<Dtype>(CblasTrans, dim, 1, 1., bottom[0]->cpu_data(), identity.cpu_data(), 0., mean_s.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, dim, 1, 1., bottom[1]->cpu_data(), identity.cpu_data(), 0., mean_t.mutable_cpu_data());
  // calculate the squared mean
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, dim, dim, 1, 1., mean_s.cpu_data(), mean_s.cpu_data() , 0., square_mean_s.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, dim, dim, 1, 1., mean_t.cpu_data(), mean_t.cpu_data() , 0., square_mean_t.mutable_cpu_data());
  // divide squared mean by (num*(num-1))
  caffe_scal(size_cov, Dtype(1./(num*(num-1))), square_mean_s.mutable_cpu_data());
  caffe_scal(size_cov, Dtype(1./(num*(num-1))), square_mean_t.mutable_cpu_data());
  //cov is (1/(num-1))*(D'*D) - (1/(num*(num-1)))*(mean)^T*(mean)
  caffe_sub(size_cov, cov_s.cpu_data(), square_mean_s.cpu_data(), cov_s.mutable_cpu_data());
  caffe_sub(size_cov, cov_t.cpu_data(), square_mean_t.cpu_data(), cov_t.mutable_cpu_data());
  //cov_s - cov_t
  caffe_sub(size_cov, cov_s.cpu_data(), cov_t.cpu_data(), diff_.mutable_cpu_data());

  Dtype dot = caffe_cpu_dot(size_cov, diff_.cpu_data(), diff_.cpu_data());
  //loss = (1/4)*(1/(dim*dim))*dot
  Dtype loss = dot / Dtype(4. * dim * dim);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CORALLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = count / num;
  // using chain rule to calculate gradients
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num, dim, 1, (1./num), identity.cpu_data(), mean_s.cpu_data(), 0., bp_mean_s.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num, dim, 1, (1./num), identity.cpu_data(), mean_t.cpu_data(), 0., bp_mean_t.mutable_cpu_data());
  // calculate bp_der_s and bp_der_t
  caffe_sub(count, bottom[0]->cpu_data(), bp_mean_s.cpu_data(), bp_der_s.mutable_cpu_data());
  caffe_sub(count, bottom[1]->cpu_data(), bp_mean_t.cpu_data(), bp_der_t.mutable_cpu_data());
  //Calculate diff_data
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, dim, (1./(dim*dim*(num-1))), bp_der_s.cpu_data(), diff_.cpu_data(), 0., diff_data_s.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, dim, (1./(dim*dim*(num-1))), bp_der_t.cpu_data(), diff_.cpu_data(), 0., diff_data_t.mutable_cpu_data());

  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0];
      if(i==0){
        caffe_cpu_axpby(
            bottom[i]->count(),              // count
            alpha,                              // alpha
            diff_data_s.cpu_data(),                   // a
            Dtype(0),                           // beta
            bottom[i]->mutable_cpu_diff());  // b
      }
      else{
        caffe_cpu_axpby(
            bottom[i]->count(),              // count
            alpha,                              // alpha
            diff_data_t.cpu_data(),                   // a
            Dtype(0),                           // beta
            bottom[i]->mutable_cpu_diff());  // b
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CORALLossLayer);
#endif

INSTANTIATE_CLASS(CORALLossLayer);
REGISTER_LAYER_CLASS(CORALLoss);

}  // namespace caffe
