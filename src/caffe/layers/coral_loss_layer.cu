#include <vector>

#include "caffe/layers/coral_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CORALLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
      const int count = bottom[0]->count();
      const int num = bottom[0]->num();
      const int dim = count / num;
      const int size_cov = dim * dim;
      // calculating D'D for source and target
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, dim, dim, num, 1., bottom[0]->gpu_data(), bottom[0]->gpu_data(), 0., cov_s.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, dim, dim, num, 1., bottom[1]->gpu_data(), bottom[1]->gpu_data(), 0., cov_t.mutable_gpu_data());
      // divide D'D by (num-1)
      caffe_gpu_scal<Dtype>(size_cov, Dtype(1./(num-1)), cov_s.mutable_gpu_data());
      caffe_gpu_scal<Dtype>(size_cov, Dtype(1./(num-1)), cov_t.mutable_gpu_data());
      // identity is a row vector of 1s
      caffe_gpu_set(num, Dtype(1), identity.mutable_gpu_data());
      // calculate the mean of D per column
      caffe_gpu_gemv<Dtype>(CblasTrans, dim, 1, 1., bottom[0]->gpu_data(), identity.gpu_data(), 0., mean_s.mutable_gpu_data());
      caffe_gpu_gemv<Dtype>(CblasTrans, dim, 1, 1., bottom[1]->gpu_data(), identity.gpu_data(), 0., mean_t.mutable_gpu_data());
      // calculate the squared mean
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, dim, dim, 1, 1., mean_s.gpu_data(), mean_s.gpu_data(), 0., square_mean_s.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, dim, dim, 1, 1., mean_t.gpu_data(), mean_t.gpu_data(), 0., square_mean_t.mutable_gpu_data());
      // divide squared mean by (num*(num-1))
      caffe_gpu_scal(size_cov, Dtype(1./(num*(num-1))), square_mean_s.mutable_gpu_data());
      caffe_gpu_scal(size_cov, Dtype(1./(num*(num-1))), square_mean_t.mutable_gpu_data());
      //cov is (1/(num-1))*(D'*D) - (1/(num*(num-1)))*(mean)^T*(mean)
      caffe_gpu_sub(size_cov, cov_s.gpu_data(), square_mean_s.gpu_data(), cov_s.mutable_gpu_data());
      caffe_gpu_sub(size_cov, cov_t.gpu_data(), square_mean_t.gpu_data(), cov_t.mutable_gpu_data());
      //cov_s - cov_t
      caffe_gpu_sub(size_cov, cov_s.gpu_data(), cov_t.gpu_data(), diff_.mutable_gpu_data());

      Dtype dot;
      caffe_gpu_dot(size_cov, diff_.gpu_data(), diff_.gpu_data(), &dot);
      //loss = (1/4)*(1/(dim*dim))*dot
      Dtype loss = dot / Dtype(4. * dim * dim);
      top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CORALLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      int count = bottom[0]->count();
      int num = bottom[0]->num();
      int dim = count / num;
      // using chain rule to calculate gradients
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num, dim, 1, (1./num), identity.gpu_data(), mean_s.gpu_data(), 0., bp_mean_s.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num, dim, 1, (1./num), identity.gpu_data(), mean_t.gpu_data(), 0., bp_mean_t.mutable_gpu_data());
      // calculate bp_der_s and bp_der_t
      caffe_gpu_sub(count, bottom[0]->gpu_data(), bp_mean_s.gpu_data(), bp_der_s.mutable_gpu_data());
      caffe_gpu_sub(count, bottom[1]->gpu_data(), bp_mean_t.gpu_data(), bp_der_t.mutable_gpu_data());
      //Calculate diff_data
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, dim, (1./(dim*dim*(num-1))), bp_der_s.gpu_data(), diff_.gpu_data(), 0., diff_data_s.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, dim, (1./(dim*dim*(num-1))), bp_der_t.gpu_data(), diff_.gpu_data(), 0., diff_data_t.mutable_gpu_data());

      for (int i = 0; i < 2; ++i) {
        if (propagate_down[i]) {
          const Dtype sign = (i == 0) ? 1 : -1;
          const Dtype alpha = sign * top[0]->cpu_diff()[0];
          if(i==0){
            caffe_gpu_axpby(
                bottom[i]->count(),              // count
                alpha,                              // alpha
                diff_data_s.gpu_data(),                   // a
                Dtype(0),                           // beta
                bottom[i]->mutable_gpu_diff());  // b
          }
          else{
            caffe_gpu_axpby(
                bottom[i]->count(),              // count
                alpha,                              // alpha
                diff_data_t.gpu_data(),                   // a
                Dtype(0),                           // beta
                bottom[i]->mutable_gpu_diff());  // b
          }
        }
      }
}

INSTANTIATE_LAYER_GPU_FUNCS(CORALLossLayer);

}  // namespace caffe
