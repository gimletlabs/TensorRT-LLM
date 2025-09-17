#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_mixed_input_launcher.inl"
namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<256>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<256>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<256>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<256>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<512>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<512>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<512>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<512>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<256>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<256>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<256>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<256>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<512>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<512>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<512>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<512>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<256>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<256>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<256>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<256>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<512>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<512>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<512>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<512>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<256>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<256>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<256>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<256>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<512>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<512>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<512>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, half,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<512>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<256>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<256>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<256>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<256>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<512>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<512>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<512>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<16>, cute::Int<512>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<256>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<256>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<256>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<256>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<512>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<512>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<512>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<32>, cute::Int<512>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<256>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<256>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<256>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<256>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<512>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<512>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<512>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<512>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<128>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<256>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<256>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<256>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<256>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<512>>, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<512>>, cute::Shape<cute::Int<1>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<512>>, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


template void sm90_generic_mixed_moe_gemm_kernelLauncher<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16,
tensorrt_llm::cutlass_extensions::EpilogueOpDefault, cute::Shape<cute::Int<64>, cute::Int<128>, cute::Int<512>>, cute::Shape<cute::Int<2>, cute::Int<2>, cute::Int<1>>, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY> (
GroupedGemmInput<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size);


} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
