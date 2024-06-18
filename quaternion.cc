#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <random>

#define CHECK_HIP(x) \
{\
  hipError_t error = (x); \
  if (error != 0) { \
  std::cerr << __LINE__ << ": HIP call failed:" #x << " error = " << error << "\n"; \
  } \
}

struct Quaternion {
  float w;
  float x;
  float y;
  float z;
};

std::mt19937 generator;

float RandomFloat(float min_value, float max_value) {
  float value = generator();
  value = (value - static_cast<float>(generator.min())) / (1.0f *
    (static_cast<float>(generator.max()) -
     static_cast<float>(generator.min())));
  return value;
}

Quaternion RandomUnitQuaternion() {
  float cos_t =  RandomFloat(-1.0f, 1.0f);
  float sin_t = std::sqrt(1.0f - cos_t * cos_t);
  float w = cos_t;
  float x = RandomFloat(0.0f, 1.0f);
  float y = RandomFloat(0.0f, 1.0f);
  float z = RandomFloat(0.0f, 1.0f);
  float mag = std::sqrt(x * x + y * y + z * z);
  float coeff = sin_t / mag;
  x *= coeff;
  y *= coeff;
  z *= coeff;
  Quaternion result{w, x, y, z};
  return result;
}

__host__ __device__ void Mul(Quaternion* q, Quaternion* r, Quaternion* result) {
  result->x = r->w * q->w - r->x * q->x - r->y * q->y - r->z * q->z;
  result->y = r->w * q->x + r->x * q->w - r->y * q->z + r->z * q->y;
  result->z = r->w * q->y + r->x * q->z + r->y * q->w - r->z * q->x;
  result->w = r->w * q->z - r->x * q->y + r->y * q->x + r->z * q->w;
}

__host__ __device__ float Mag(Quaternion* q) {
  return std::sqrt(q->w * q->w + q->x * q->x + q->y * q->y + q->z * q->z);
}

Quaternion Sub(const Quaternion& q, const Quaternion& r) {
  Quaternion result{q.w - r.w, q.x - r.x, q.y - r.y, q.z - r.z};
  return result;
}

void GenerateRandomQuaternions(size_t n, std::vector<Quaternion>& quaternions) {
  for (size_t i = 0; i < n; ++i) {
    quaternions.push_back(RandomUnitQuaternion());
  }
}

void MultiplyQuaternionsCpu(size_t num_quaternions, Quaternion* left,
                            Quaternion* right, Quaternion* result) {
  Quaternion* current_left = left;
  Quaternion* current_right = right;
  Quaternion* current_result = result;
  for (size_t i = 0; i < num_quaternions; ++i) {
    Mul(current_left, current_right, current_result);
    ++current_left;
    ++current_right;
    ++current_result;
  }
}

__global__ void MultiplyQuaternionsGpu(size_t num_quaternions, Quaternion* left,
    Quaternion* right, Quaternion* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_quaternions) {
      Mul(&left[tid], &right[tid], &result[tid]);
    }
}

int main(int argc, char** argv) {
  size_t num_quaternions = 1024 * 1024; 
  std::vector<Quaternion> left;
  std::vector<Quaternion> right;
  size_t num_bytes = sizeof(Quaternion) * num_quaternions;
  auto start_init = std::chrono::system_clock::now();
  GenerateRandomQuaternions(num_quaternions, left);
  GenerateRandomQuaternions(num_quaternions, right);
  auto end_init = std::chrono::system_clock::now();
  std::chrono::milliseconds duration_init =
    std::chrono::duration_cast<std::chrono::milliseconds>(end_init - start_init);
  std::cout << "[INIT] " << "Initialization took " <<
    duration_init.count() << " ms.\n";
  std::cout << "[INIT] " << "Using "
            << (3.0f * num_bytes) / (1000*1000*1000) << " GB\n";

  std::unique_ptr<Quaternion[]> result_cpu =
    std::make_unique<Quaternion[]>(num_quaternions);

  auto start_cpu = std::chrono::system_clock::now();
  MultiplyQuaternionsCpu(num_quaternions, &left[0], &right[0], result_cpu.get());
  auto end_cpu = std::chrono::system_clock::now();
  std::chrono::milliseconds duration_cpu =
    std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);
  std::cout << "[CPU] Multiplying " << num_quaternions << " quaternions took " <<
    duration_cpu.count() << " ms.\n";

  std::unique_ptr<Quaternion[]> result_gpu =
    std::make_unique<Quaternion[]>(num_quaternions);

  auto start_gpu_total = std::chrono::system_clock::now();

  Quaternion* dev_left;
  Quaternion* dev_right;
  Quaternion* dev_result;
  CHECK_HIP(hipMalloc(&dev_left, num_quaternions * sizeof(Quaternion)));
  CHECK_HIP(hipMalloc(&dev_right, num_quaternions * sizeof(Quaternion)));
  CHECK_HIP(hipMalloc(&dev_result, num_quaternions * sizeof(Quaternion)));

  CHECK_HIP(hipMemcpy(dev_left, &left[0], num_bytes, hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(dev_right, &right[0], num_bytes, hipMemcpyHostToDevice));

  size_t threads_per_block = 512;
  size_t num_blocks = (num_quaternions + threads_per_block - 1) / threads_per_block;
  dim3 threads(threads_per_block);
  dim3 blocks(num_blocks);
  auto start_gpu = std::chrono::system_clock::now();
  MultiplyQuaternionsGpu<<<blocks,threads>>>(num_quaternions, dev_left,
                                             dev_right, dev_result);

  CHECK_HIP(hipDeviceSynchronize());
  auto end_gpu = std::chrono::system_clock::now();
  std::chrono::milliseconds duration_gpu =
    std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu);
  std::cout << "[GPU] Multiplying " << num_quaternions << " quaternions took " <<
    duration_gpu.count() << " ms.\n";

  CHECK_HIP(hipMemcpy(result_gpu.get(), dev_result, num_bytes,
                      hipMemcpyDeviceToHost));

  CHECK_HIP(hipFree(dev_left));
  CHECK_HIP(hipFree(dev_right));
  CHECK_HIP(hipFree(dev_result));

  auto end_gpu_total = std::chrono::system_clock::now();
  std::chrono::milliseconds duration_gpu_total =
    std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu_total - start_gpu_total);

  std::cout << "[GPU] Total time, including HIP setup and copy took " <<
    duration_gpu_total.count() << " ms.\n";

  for (int i = 0; i < num_quaternions; ++i) {
    Quaternion diff = Sub(result_cpu[i], result_gpu[i]);
    float error = Mag(&diff);
    if (error  > 1e-5) {
      std::cout << "Error too large at index " << i << " error = "
        << error << "\n";
      break;
    }
  }

  return 0;
}






