#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include "qw3_mean_k_kernel.cuh"
#include "qw3_mean_k_tiled.cuh"

torch::Tensor score(torch::Tensor q, torch::Tensor k, double scale) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && q.device() == k.device(), "CUDA Q/index must share a device");
    TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf, "qw3 MeanK requires FP16 Q/index");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && q.is_contiguous() && k.is_contiguous(), "Expected contiguous LTHD/LBHD");
    const auto L=q.size(0), T=q.size(1), H=q.size(2), D=q.size(3), B=k.size(1), KH=k.size(2);
    TORCH_CHECK(L>0 && T>0 && H>0 && D>0 && B>0 && B<=32768 && KH>0 && H%KH==0 &&
        k.size(0)==L && k.size(3)==D && std::isfinite(scale) && scale>0, "Invalid bounded MeanK geometry/scale");
    TORCH_CHECK(L*T*H<=UINT32_MAX && D<=UINT32_MAX, "MeanK distribution dimensions exceed native indexing");
    c10::cuda::CUDAGuard guard(q.device());
    auto result=torch::zeros({B},q.options().dtype(at::kFloat));
    const auto stream=at::cuda::getCurrentCUDAStream();
    if (B<=8192) {
        // NVCC 13 / sm_120 emits a 128-bit shared load for the scalar
        // reduction's final 1-3 elements. Reserve its full tail access;
        // n_blocks stays B, so padding never participates in the softmax.
        const auto shared_bytes = ((B+3)/4)*4*sizeof(float);
        block_attn_score_softmax_pages_kernel<at::Half,at::Half><<<L*T,128,shared_bytes,stream>>>(
            result.data_ptr<float>(),q.data_ptr<at::Half>(),k.data_ptr<at::Half>(),nullptr,nullptr,0,
            L,T,T,B,B,H,KH,D,static_cast<float>(scale),0,B,1,0);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        // qw3's production large-index path: one dot per distribution/page,
        // globally normalize all pages, then accumulate bounded batches.
        const uint32_t distributions=L*T*H;
        const uint32_t batch=std::min<int64_t>(distributions,(64ll*1024*1024/sizeof(float))/B);
        auto workspace=torch::empty({int64_t(batch)*(B+2)},result.options());
        float *logits=workspace.data_ptr<float>();
        float *row_max=logits+int64_t(batch)*B;
        float *row_inv=row_max+batch;
        const uint32_t tiles=(B+kSoftmaxPagesTile-1)/kSoftmaxPagesTile;
        const float weight=1.0f/(static_cast<float>(L)*static_cast<float>(H));
        for (uint32_t base=0;base<distributions;) {
            const uint32_t count=std::min(batch,distributions-base);
            TORCH_CHECK(uint64_t(count)*tiles<=0x7fffffffu,"MeanK launch grid exceeds native indexing");
            block_attn_softmax_pages_logits_kernel<at::Half,at::Half><<<count*tiles,kSoftmaxPagesThreads,0,stream>>>(
                logits,q.data_ptr<at::Half>(),k.data_ptr<at::Half>(),base,count,tiles,T,T,B,B,H,KH,D,
                static_cast<float>(scale),0,B,1);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            block_attn_softmax_pages_norm_kernel<<<count,kSoftmaxPagesThreads,0,stream>>>(row_max,row_inv,logits,count,B);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            block_attn_softmax_pages_accumulate_logits_kernel<<<(B+kSoftmaxPagesThreads-1)/kSoftmaxPagesThreads,kSoftmaxPagesThreads,0,stream>>>(
                result.data_ptr<float>(),logits,row_max,row_inv,count,B,1,0,weight,0,B);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            base+=count;
        }
    }
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("score", &score); }
