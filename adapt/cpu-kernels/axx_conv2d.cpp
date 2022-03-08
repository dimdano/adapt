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

//for 12 bit you have to change bits in conv2d class and put appropriate pytorch dtype

//here you need to change input dtype and output dtype to not overflow (for int8 out must be 32bit)
//also change naive_sgemm temp dtypes. also change my_addmm call template arguement. and AND with appropriate bits (4095 for 12bit)

#include <immintrin.h>

namespace at {
namespace native { 
    
namespace { 
      
   
    
template <typename in_t, typename out_t >      
void naive_sgemm_simple(
 int M, int N, int K, 
 in_t alpha, 
 in_t *A, int lda, 
 in_t *B, int ldb,
 in_t beta, 
 out_t *C, int ldc)
{

    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < N; ++j) {
            out_t temp = 0;
            for(int k = 0; k < K; ++k) {
                uint8_t a = A[i*K+k];
                uint8_t b = B[k*N+j];
                temp += lut[a][b];
            }
            C[i*ldc+j] = temp;
        }
    }
}

//faster implementation with manual loop unrolls
template <typename in_t, typename out_t >      
void naive_sgemm(
 int M, int N, int K, 
 in_t alpha, 
 in_t *A, int lda, 
 in_t *B, int ldb,
 in_t beta, 
 out_t *C, int ldc)
{
    
    int b = K/8*8;

    //std::cout << M << " " << N << " " << K << std::endl;

    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < N; ++j) {
            out_t temp = 0;
            
            __attribute__(( aligned(32))) out_t temp_arr[8]={0};            

            for(int k = 0; k < b; k+=8) {
                temp_arr[0] += lut[(uint8_t)A[i*lda+k]][(uint8_t)B[j*K+k]];
                temp_arr[1] += lut[(uint8_t)A[i*lda+k+1]][(uint8_t)B[j*K+k+1]];
                temp_arr[2] += lut[(uint8_t)A[i*lda+k+2]][(uint8_t)B[j*K+k+2]];
                temp_arr[3] += lut[(uint8_t)A[i*lda+k+3]][(uint8_t)B[j*K+k+3]];
                temp_arr[4] += lut[(uint8_t)A[i*lda+k+4]][(uint8_t)B[j*K+k+4]];
                temp_arr[5] += lut[(uint8_t)A[i*lda+k+5]][(uint8_t)B[j*K+k+5]];
                temp_arr[6] += lut[(uint8_t)A[i*lda+k+6]][(uint8_t)B[j*K+k+6]];
                temp_arr[7] += lut[(uint8_t)A[i*lda+k+7]][(uint8_t)B[j*K+k+7]];
            }
            
            temp += temp_arr[0]+temp_arr[1]+temp_arr[2]+temp_arr[3]+temp_arr[4]+temp_arr[5]+temp_arr[6]+temp_arr[7];
            for(int k = b; k < K; k++) {
                temp += lut[(uint8_t)A[i*lda+k]][(uint8_t)B[j*K+k]];
            } 
            
            C[i*ldc+j] = temp; 
        }
    } 
}   
    
template <typename in_t, typename out_t>    
Tensor my_addmm(const Tensor& self, const Tensor& m1, const Tensor& m2, Scalar beta = 1, Scalar alpha = 1) {
    
  TORCH_CHECK(m1.dim() == 2, "mat1 must be a matrix, got ", m1.dim(), "-D tensor");
  TORCH_CHECK(m2.dim() == 2, "mat2 must be a matrix, got ", m2.dim(), "-D tensor"); 

  // Array access is faster than .size(n) and .stride(n)
  const auto self_sizes = self.sizes();
  //auto self_strides = self.strides(); 
  //auto m1_strides = m1.strides();
  auto m1_sizes = m1.sizes();
  //auto m2_strides = m2.strides();
  auto m2_sizes = m2.sizes();

 // Tensor result = repeat(self, {m1_sizes[0], 1});
  //self.resize_(m1_sizes[0], 1)
    
  TORCH_INTERNAL_ASSERT(self.dim() == 2 && m1.dim() == 2 && m2.dim() == 2);
  TORCH_CHECK(
      self_sizes[0] == m1_sizes[0] && self_sizes[1] == m2_sizes[1],
      "input shape is incompatible with matrix multiplication (",
      m1_sizes[0], "x", m1_sizes[1], " @ ", m2_sizes[0], "x", m2_sizes[1], " != ",
      self_sizes[0], "x", self_sizes[1], ")");
              

  if (self.numel() == 0) {
    return self;
  }


  auto m = m1_sizes[0];
  auto k = m1_sizes[1];
  auto n = m2_sizes[1];
    
  const int lda = k;
  const int ldb = n;
  const int ldc = n;
  
  //transpose b matrix - better cache re-use achieves slightly higher performance
  Tensor m2_t;
  m2_t = (m2.transpose(0, 1)).contiguous();
   
    //call sgemm kernel
    naive_sgemm <in_t, out_t> (m, n, k, alpha.to<in_t>(), m1.data_ptr<in_t>(), lda, m2_t.data_ptr<in_t>(), ldb, beta.to<in_t>(), self.data_ptr<out_t>(), ldc);

    return self;
     
}     
    
    
template <typename scalar_t>
static void unfolded2d_copy1(
    scalar_t* input_data,
    scalar_t* finput_data,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {
  at::parallel_for(
      0, (int64_t)n_input_plane * kH * kW, 0, [&](int64_t start, int64_t end) {
        for (auto k = start; k < end; k++) {
          int64_t nip = k / (kH * kW);
          int64_t rest = k % (kH * kW);
          int64_t kh = rest / kW;
          int64_t kw = rest % kW;
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          int64_t x, y;
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          int64_t ix, iy;
          scalar_t* dst = finput_data +
              nip * ((size_t)kH * kW * output_height * output_width) +
              kh * ((size_t)kW * output_height * output_width) +
              kw * ((size_t)output_height * output_width);
          scalar_t* src =
              input_data + nip * ((size_t)input_height * input_width);
          if (padW > 0 || padH > 0) {
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            int64_t lpad, rpad;
            for (y = 0; y < output_height; y++) {
              iy = (int64_t)y * dH - padH + kh;
              if (iy < 0 || iy >= input_height) {
                memset(
                    dst + (size_t)y * output_width,
                    0,
                    sizeof(scalar_t) * output_width);
              } else {
                if (dW == 1) {
                  ix = 0 - padW + kw;
                  lpad = std::max<int64_t>(0, padW - kw);
                  rpad = std::max<int64_t>(0, padW - (kW - kw - 1));
                  if (output_width - rpad - lpad <= 0) {
                    memset(
                        dst + (size_t)y * output_width,
                        0,
                        sizeof(scalar_t) * output_width);
                  } else {
                    if (lpad > 0)
                      memset(
                          dst + (size_t)y * output_width,
                          0,
                          sizeof(scalar_t) * lpad);
                    memcpy(
                        dst + (size_t)y * output_width + lpad,
                        src + (size_t)iy * input_width + ix + lpad,
                        sizeof(scalar_t) * (output_width - rpad - lpad));
                    if (rpad > 0)
                      memset(
                          dst + (size_t)y * output_width + output_width - rpad,
                          0,
                          sizeof(scalar_t) * rpad);
                  }
                } else {
                  for (x = 0; x < output_width; x++) {
                    ix = (int64_t)x * dW - padW + kw;
                    if (ix < 0 || ix >= input_width)
                      memset(
                          dst + (size_t)y * output_width + x,
                          0,
                          sizeof(scalar_t) * 1);
                    else
                      memcpy(
                          dst + (size_t)y * output_width + x,
                          src + (size_t)iy * input_width + ix,
                          sizeof(scalar_t) * (1));
                  }
                }
              }
            }
          } else {
            for (y = 0; y < output_height; y++) {
              iy = (int64_t)y * dH + kh;
              ix = 0 + kw;
              if (dW == 1)
                memcpy(
                    dst + (size_t)y * output_width,
                    src + (size_t)iy * input_width + ix,
                    sizeof(scalar_t) * output_width);
              else {
                for (x = 0; x < output_width; x++)
                  memcpy(
                      dst + (size_t)y * output_width + x,
                      src + (size_t)iy * input_width + ix + (int64_t)x * dW,
                      sizeof(scalar_t) * (1));
              }
            }
          }
        }
      });
}

void unfolded2d_copy_stub1(
    Tensor& finput,
    Tensor& input,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {
  // This function assumes that
  // kH*kW does not overflow an int
  // n_input_plane*kH*kW does not overflow a int64_t
  // output_height*dH does not overflow a int64_t
  // output_width*dW does not overflow a int64_t


  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::BFloat16, input.scalar_type(), "unfolded2d_copy", [&] {
        scalar_t* input_data = input.data_ptr<scalar_t>();
        scalar_t* finput_data = finput.data_ptr<scalar_t>();

        unfolded2d_copy1(
            input_data,
            finput_data,
            kH,
            kW,
            dH,
            dW,
            padH,
            padW,
            n_input_plane,
            input_height,
            input_width,
            output_height,
            output_width);
      });
}
    
    
    
    
static inline void slow_conv2d_shape_check(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    const Tensor& bias,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    bool weight_optional) {
  TORCH_CHECK(
      kernel_width > 0 && kernel_height > 0,
      "kernel size should be greater than zero, but got kernel_height: ",
      kernel_height,
      " kernel_width: ",
      kernel_width);
  TORCH_CHECK(
      stride_width > 0 && stride_height > 0,
      "stride should be greater than zero, but got stride_height: ",
      stride_height,
      " stride_width: ",
      stride_width);

  if (weight.defined()) {
    TORCH_CHECK(
        weight.numel() > 0 && (weight.dim() == 2 || weight.dim() == 4),
        "non-empty 2D or 4D weight tensor expected, but got: ",
        weight.sizes());
    if (bias.defined()) {
      check_dim_size(bias, 1, 0, weight.size(0));
    }
  } else {
    TORCH_CHECK(weight_optional, "weight tensor is undefined");
  }

  const int64_t ndim = input.dim();
  const int64_t dim_batch = 0;
  const int64_t dim_planes = 1;
  const int64_t dim_height = 2;
  const int64_t dim_width = 3;

  // Allow for empty batch size but not other dimensions
  bool valid_empty = ndim == 4 && input.size(dim_batch) == 0 &&
      input.size(dim_planes) != 0 && input.size(dim_height) != 0 &&
      input.size(dim_width) != 0;

  TORCH_CHECK(
      (input.numel() > 0 || valid_empty) && ndim == 4,
      "non-empty 4D input tensor expected but got: ",
      input.sizes());

  const int64_t input_height = input.size(dim_height);
  const int64_t input_width = input.size(dim_width);

  const int64_t exact_input_height = input_height + 2 * pad_height;
  const int64_t exact_input_width = input_width + 2 * pad_width;

  TORCH_CHECK(
      exact_input_height >= kernel_height && exact_input_width >= kernel_width,
      "Calculated padded input size per channel: (",
      exact_input_height,
      " x ",
      exact_input_width,
      "). ",
      "Kernel size: (",
      kernel_height,
      " x ",
      kernel_width,
      "). Kernel size can't be greater than actual input size");

  const int64_t output_height =
      div_rtn<int64_t>(exact_input_height - kernel_height, stride_height) + 1;
  const int64_t output_width =
      div_rtn<int64_t>(exact_input_width - kernel_width, stride_width) + 1;

  TORCH_CHECK(
      output_width >= 1 && output_height >= 1,
      "Given input size per channel: (",
      input_height,
      " x ",
      input_width,
      "). "
      "Calculated output size per channel: (",
      output_height,
      " x ",
      output_width,
      "). Output size is too small");

  if (weight.defined()) {
    int64_t n_input_plane = weight.size(1);
    if (weight.dim() == 2) {
      n_input_plane /= (kernel_height * kernel_width);
    }
    check_dim_size(input, ndim, dim_planes, n_input_plane);
  }

  if (grad_output.defined()) {
    if (weight.defined()) {
      int64_t n_output_plane = weight.size(0);
      check_dim_size(grad_output, ndim, dim_planes, n_output_plane);
    } else if (bias.defined()) {
      TORCH_CHECK(bias.numel() > 0, "non-empty bias tensor expected");
      const int64_t n_output_plane = bias.dim() == 0 ? 1 : bias.size(0);
      check_dim_size(grad_output, ndim, dim_planes, n_output_plane);
    }
    check_dim_size(grad_output, ndim, dim_height, output_height);
    check_dim_size(grad_output, ndim, dim_width, output_width);
  }
}

static Tensor view_weight_2d(const Tensor& weight_) {
  Tensor weight = weight_.contiguous();
  if (weight.dim() == 4) {
    const int64_t s1 = weight.size(0);
    const int64_t s2 = weight.size(1) * weight.size(2) * weight.size(3);
    return weight.view({s1, s2});
  } else {
    return weight;
  }
}

static void slow_conv2d_update_output_frame(
    Tensor& input,
    Tensor& output,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& finput,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t n_input_plane,
    int64_t input_height,
    int64_t input_width,
    int64_t n_output_plane,
    int64_t output_height,
    int64_t output_width) {
  unfolded2d_copy_stub1(
      finput,
      input,
      kernel_height,
      kernel_width,
      stride_height,
      stride_width,
      pad_height,
      pad_width,
      n_input_plane,
      input_height,
      input_width,
      output_height,
      output_width);

  auto output2d =
      output.reshape({n_output_plane, output_height * output_width});//.to(at::ScalarType::Int);

  if (bias.defined()) {
    for (int64_t i = 0; i < n_output_plane; i++) {
      output[i].fill_(bias[i].item());
    }
  } else {
    output.zero_();
  }
              
   //int* input_data = input.data_ptr<scalar_t>();       
    //TODO: need to fix dispatch when working with bitwise logic in lut (cause it dispatches float and double funcs)
  my_addmm <int8_t, int> (output2d, weight, finput, 0, 1); 
   //   });
    
   
    
}




} // namespace

std::tuple<Tensor&, Tensor&, Tensor&> slow_conv2d_forward_out_cpu(
    Tensor& output,
    Tensor& finput,
    Tensor& fgrad_input,
    const Tensor& self,
    const Tensor& weight_,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding) {
  const int64_t kernel_height = kernel_size[0];
  const int64_t kernel_width = kernel_size[1];
  const int64_t pad_height = padding[0];
  const int64_t pad_width = padding[1];
  const int64_t stride_height = stride[0];
  const int64_t stride_width = stride[1];

  const Tensor weight_2d = view_weight_2d(weight_);

  slow_conv2d_shape_check(
      self,
      Tensor(),
      weight_2d,
      bias,
      kernel_height,
      kernel_width,
      stride_height,
      stride_width,
      pad_height,
      pad_width,
      false);

  const Tensor input = self.contiguous();
  //const int64_t ndim = input.dim();
  const int64_t dim_planes = 1;
  const int64_t dim_height = 2;
  const int64_t dim_width = 3;

  const int64_t n_input_plane = input.size(dim_planes);
  const int64_t input_height = input.size(dim_height);
  const int64_t input_width = input.size(dim_width);
  const int64_t n_output_plane = weight_2d.size(0);
  const int64_t output_height =
      (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
  const int64_t output_width =
      (input_width + 2 * pad_width - kernel_width) / stride_width + 1;

  const int64_t batch_size = input.size(0);

  finput.resize_({batch_size,
                  n_input_plane * kernel_height * kernel_width,
                  output_height * output_width});
  output.resize_({batch_size, n_output_plane, output_height, output_width});

  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    NoGradGuard no_grad;
    AutoNonVariableTypeMode non_variable_type_mode;
      
    for (int64_t t = start; t < end; t++) {
      Tensor input_t = input[t];
      Tensor output_t = output[t];
      Tensor finput_t = finput[t];
      slow_conv2d_update_output_frame(
          input_t,
          output_t,
          weight_2d,
          bias,
          finput_t,
          kernel_height,
          kernel_width,
          stride_height,
          stride_width,
          pad_height,
          pad_width,
          n_input_plane,
          input_height,
          input_width,
          n_output_plane,
          output_height,
          output_width);
    }
  });

  return std::tuple<Tensor&, Tensor&, Tensor&>(output, finput, fgrad_input);
}
    
    
//only works for no bias at the moment - add bias on python level 
Tensor slow_axx_conv2d_forward_cpu(
    const Tensor& self,
    const Tensor& weight,
    c10::IntArrayRef kernel_size,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding) {
    
  const Tensor& bias = {};   //put empty bias here for no-bias cases. when put in function arguements throws error
    
  //output type must change at start  
  auto output = at::zeros({0}, at::ScalarType::Int);
         
  auto finput = at::empty({0}, self.options());
  auto fgrad_input = at::empty({0}, self.options());
  slow_conv2d_forward_out_cpu(
      output,
      finput,
      fgrad_input,
      self,
      weight,
      kernel_size,
      bias,
      stride,
      padding);
    
    return output;
}

} // namespace native
} // namespace at



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &at::native::slow_axx_conv2d_forward_cpu, "axx_conv2d");
}
                
