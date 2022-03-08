#include <torch/extension.h>
#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/grad_mode.h>
#include <ATen/div_rtn.h>
#include <ATen/native/Unfold2d.h>

//include header file with approx-mult LUT table 
//passed by preprocessor flag using jit torch load 
//the header file must include a constant 2d array named lut
#define STR(s) STR2(s)
#define STR2(s) #s
#define EXPAND(s) s

#include STR(axx_mults/EXPAND(AXX_MULT).h)

    
namespace at {
namespace native {
 
    
template <typename in_t, typename out_t >
void naive_sgemm(
 int M, int N, int K, 
 in_t alpha, 
 in_t *A, int lda, 
 in_t *B, int ldb,
 in_t beta, 
 out_t *C, int ldc)
{
        
    at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) { 
    for(int i = begin; i < end; ++i) {
        for(int j = 0; j < N; ++j) {
            out_t temp = 0;
            for(int k = 0; k < K; ++k) {
                uint8_t a = A[i*lda+k];
                uint8_t b = B[j*K+k];
                temp += lut[a][b];
             }
            C[i*ldc+j] = temp; 
        }
    }
});

    
}


Tensor my_addmm(const Tensor& result, const Tensor& m1, const Tensor& m2, Scalar beta = 1, Scalar alpha = 1) {
    //Tensor result = at::empty({0}, self.options());
    //return addmm_cpu_out(result, self, mat1, mat2, beta, alpha);
    
  TORCH_CHECK(m1.dim() == 2, "mat1 must be a matrix, got ", m1.dim(), "-D tensor");
  TORCH_CHECK(m2.dim() == 2, "mat2 must be a matrix, got ", m2.dim(), "-D tensor");
 

  // Array access is faster than .size(n) and .stride(n)
  const auto self_sizes = result.sizes();
  auto m1_strides = m1.strides();
  auto m1_sizes = m1.sizes();
  auto m2_strides = m2.strides();
  auto m2_sizes = m2.sizes();

  auto result_sizes = result.sizes(); 
  auto result_strides = result.strides(); 
  

  TORCH_INTERNAL_ASSERT(result.dim() == 2 && m1.dim() == 2 && m2.dim() == 2);
    
  TORCH_CHECK(
      m1_sizes[1] == m2_sizes[0], "mat1 and mat2 shapes cannot be multiplied (",
      m1_sizes[0], "x", m1_sizes[1], " and ", m2_sizes[0], "x", m2_sizes[1], ")");

 TORCH_CHECK(
      result_sizes[0] == m1_sizes[0] && result_sizes[1] == m2_sizes[1],
      "input shape is incompatible with matrix multiplication (",
      m1_sizes[0], "x", m1_sizes[1], " @ ", m2_sizes[0], "x", m2_sizes[1], " != ",
      result_sizes[0], "x", result_sizes[1], ")");

  //native::resize_(result, self_sizes);
        
  if (result.numel() == 0) {
    return result;
  }

  bool transpose_a = false, transpose_b = false, transpose_c = false;
    

    auto m = m1_sizes[0];
    auto k = m1_sizes[1];
    auto n = m2_sizes[1];
    
  const int64_t lda = k;
  const int64_t ldb = n;
  const int64_t ldc = n;
    
 
       naive_sgemm <int8_t, int> (m, n, k, alpha.to<int8_t>(), m1.data_ptr<int8_t>(), lda, m2.data_ptr<int8_t>(), ldb, beta.to<int8_t>(), result.data_ptr<int>(), ldc);   
    
     return result;
} 
    
  
Tensor dense_forward( 
     const Tensor& input, const Tensor& weights) {

     if (input.dim() == 2) {  
         
         //accumulator is integer type
         auto result = at::zeros({input.sizes()[0], weights.sizes()[0]}, at::ScalarType::Int);    
         
         //call custom multiplication function (weights are transposed)
         //in case of bias we add it in Python class
         return my_addmm(result, input, weights.t());    
         
    }else{
        auto output = input.matmul(weights.t());
        return output;
    }
  
}

    
//TO_DO backward
Tensor dense_backward( 
        const Tensor& input, const Tensor& weights, const Tensor& bias=Tensor()) {
    
    Tensor out = bias.clone(at::MemoryFormat::Contiguous);

    return out;
}
    
    
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &at::native::dense_forward, "dense_fw");
    m.def("backward", &at::native::dense_backward, "dense_bw");
}
