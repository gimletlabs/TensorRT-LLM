#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/launchers/fpA_intB_launcher_sm90.inl"
namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_fp8_e4m3*, const cutlass::uint4b_t*, const half*, const half*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, cutlass::uint4b_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const cutlass::uint4b_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const cutlass::uint4b_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, uint8_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const uint8_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, uint8_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const uint8_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, uint8_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const uint8_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, uint8_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const uint8_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, uint8_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const uint8_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, uint8_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const uint8_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, uint8_t, half, half, half,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const uint8_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, uint8_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const uint8_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, uint8_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const uint8_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, uint8_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const uint8_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, uint8_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const uint8_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, uint8_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const uint8_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, uint8_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const uint8_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, uint8_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const uint8_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, uint8_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const uint8_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, uint8_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const uint8_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, uint8_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const uint8_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, uint8_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const uint8_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, uint8_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const uint8_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, uint8_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const uint8_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<half, uint8_t, half, half, half,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const half*, const uint8_t*, const half*, const half*, const half*, const float,
half*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, uint8_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, uint8_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, uint8_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, uint8_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, uint8_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, uint8_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, uint8_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, uint8_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, uint8_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, uint8_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, uint8_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, uint8_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, uint8_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, uint8_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, uint8_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, uint8_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, uint8_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, uint8_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, uint8_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, uint8_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


template void sm90_generic_mixed_gemm_kernelLauncher<__nv_bfloat16, uint8_t, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, tensorrt_llm::cutlass_extensions::EpilogueOpBias,
cute::Shape<cute::Int<64>, cute::Int<256>, cute::Int<64>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>,
cutlass::gemm::KernelTmaWarpSpecializedPingpongMixedInput, cutlass::epilogue::TmaWarpSpecialized> (
const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float,
__nv_bfloat16*, int, int, int, const int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig, char*, size_t, cudaStream_t, int*
);


} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
