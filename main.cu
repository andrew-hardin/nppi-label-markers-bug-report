#include <cstdint>
#include <iostream>
#include <vector>

#include <nppi.h>

#define CHECK_CUDA(x) \
  if((x) != cudaSuccess) exit(1)

#define CHECK_NPP(x) \
  if((x) != NPP_SUCCESS) exit(1)

static const int kWidth = 6;
static const int kHeight = 7;
static const NppiSize kSize = { kWidth, kHeight };

template<typename T>
void PrintGrid(const std::vector<T>& item) {
  size_t i = 0;
  for(int r = 0; r < kHeight; r++) {
    for(int c = 0; c < kWidth; c++) {
      std::cout << static_cast<int>(item[i++]) << ' ';
    }
    std::cout << '\n';
  }
}

int main() {

  std::vector<uint8_t> input_host = {
        0, 1, 1, 1, 0, 0,
        0, 1, 1, 1, 0, 0,
        0, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 1, 1,
        1, 1, 0, 0, 1, 1,
        1, 1, 0, 0, 1, 1,
  };

  // Input -> Device.
  void* input_device;
  CHECK_CUDA(cudaMalloc(&input_device, sizeof(uint8_t) * input_host.size()));
  CHECK_CUDA(cudaMemcpy(input_device, input_host.data(), sizeof(uint8_t) * input_host.size(), cudaMemcpyKind::cudaMemcpyHostToDevice));

  // Output.
  void* output_device;
  CHECK_CUDA(cudaMalloc(&output_device, sizeof(uint32_t) * input_host.size()));

  // Tmp buffer.
  int tmp_buffer_size;
  CHECK_NPP(nppiLabelMarkersUFGetBufferSize_32u_C1R(kSize, &tmp_buffer_size));
  void* tmp_buffer;
  CHECK_CUDA(cudaMalloc(&tmp_buffer, tmp_buffer_size));

  // Invoke.
  CHECK_NPP(nppiLabelMarkersUF_8u32u_C1R(
    static_cast<Npp8u*>(input_device), kWidth,
    static_cast<Npp32u*>(output_device), kWidth * sizeof(uint32_t),
    kSize, NppiNorm::nppiNormL1,
    static_cast<Npp8u*>(tmp_buffer)));

  // Copy result back to host.
  std::vector<uint32_t> output_host(input_host.size());
  CHECK_CUDA(cudaMemcpy(output_host.data(), output_device, sizeof(uint32_t) * output_host.size(), cudaMemcpyKind::cudaMemcpyDeviceToHost));

  std::cout << "INPUT\n";
  std::cout << "-----\n";
  PrintGrid(input_host);

  std::cout << "\nOUTPUT\n";
  std::cout << "-----\n";
  PrintGrid(output_host);

  return 0;
}