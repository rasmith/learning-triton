import torch

import triton
import triton.language as tl

import time

def mul_op(p, q):
    result = torch.empty_like(p)
    r = p[0]
    v = p[1:4]
    s = q[0]
    w = q[1:4]
    result[0] = r * s - v.dot(w)
    result[1:4] = r * w + s * v - torch.cross(v, w, dim = 0)
    return result

@triton.jit 
def mul_kernel_loop(p_ptr,   # pointer to first input quaternion
               q_ptr,  # pointer to first input quaternion
               result_ptr, # pointer to output quaternions
               num_quaternions, # number of quaternions
               block_size: tl.constexpr # number of assigned elements to process
               ):
    pid = tl.program_id(axis = 0) # 1D launch grid, axis = 0.
    block_start = pid * block_size # start of block

    num_cols = 4
    num_rows = block_size // 4
    p_current = p_ptr + block_start
    q_current = q_ptr + block_start
    result_current = result_ptr + block_start 
    for i in tl.range(0, block_size // num_cols):
        r = tl.load(p_current + 0)
        v0 = tl.load(p_current + 1)
        v1 = tl.load(p_current + 2)
        v2 = tl.load(p_current + 3)
        s = tl.load(q_current + 0)
        w0 = tl.load(q_current + 1)
        w1 = tl.load(q_current + 2)
        w2 = tl.load(q_current + 3)
        tl.store(result_current + 0, 
            r * s - v0 * v0 - v1 * v1 - v2 * v2)
        tl.store(result_current + 1,
            r * w0 + s * v0 - v1 * w2 +  v2 * w1)
        tl.store(result_current + 2,
            r * w1 + s * v1 - v2 * w0 + v0 * w2)
        tl.store(result_current + 3,
            r * w2 + s * v2 - v0 * w1 + v1 * w0)
        p_current += num_cols
        q_current += num_cols
        result_current += num_cols

@triton.jit 
def mul_kernel_no_loop(p_ptr,   # pointer to first input quaternion
               q_ptr,  # pointer to first input quaternion
               result_ptr, # pointer to output quaternions
               num_quaternions, # number of quaternions
               block_size: tl.constexpr # number of assigned elements to process
               ):
    pid = tl.program_id(axis = 0) # 1D launch grid, axis = 0.
    block_start = pid * block_size # start of block

    # Compute the scalar part.
    offsets = block_start + 4 * tl.arange(0, block_size // 4)

    r = tl.load(p_ptr + offsets)
    s = tl.load(q_ptr + offsets)
    v0 = tl.load(p_ptr + offsets + 1)
    v1 = tl.load(p_ptr + offsets + 2)
    v2 = tl.load(p_ptr + offsets + 3)
    w0 = tl.load(q_ptr + offsets + 1)
    w1 = tl.load(q_ptr + offsets + 2)
    w2 = tl.load(q_ptr + offsets + 3)

    # Compute the vector part.
    tl.store(result_ptr + offsets + 0,
             r * s - v0 * w0 - v1 * w1 - v2 * w2)
    tl.store(result_ptr + offsets + 1,
             r * w0 + s * v0 - v1 * w2 +  v2 * w1)
    tl.store(result_ptr + offsets + 2,
             r * w1 + s * v1 - v2 * w0 + v0 * w2)
    tl.store(result_ptr + offsets + 3,
             r * w2 + s * v2 - v0 * w1 + v1 * w0)

def mul_loop(q: torch.Tensor, r: torch.Tensor, result: torch.Tensor):
    assert q.is_cuda and r.is_cuda and result.is_cuda
    num_quaternions = result.numel()
    grid = lambda meta: (triton.cdiv(num_quaternions, meta['block_size']), )
    mul_kernel_loop[grid](q, r, result, num_quaternions, block_size = 512)
    return result

def mul_no_loop(q: torch.Tensor, r: torch.Tensor, result: torch.Tensor):
    assert q.is_cuda and r.is_cuda and result.is_cuda
    num_quaternions = result.numel()
    grid = lambda meta: (triton.cdiv(num_quaternions, meta['block_size']), )
    mul_kernel_no_loop[grid](q, r, result, num_quaternions, block_size = 512)
    return result

torch.manual_seed(0)
size = (1024 * 1024, 4)
left = torch.rand(size, device='cuda', dtype=torch.float32)
right = torch.rand(size, device='cuda', dtype=torch.float32)

compare_cpu = False

print(f"p[0:4] = {left[0:4]}")
print(f"q[0:4] = {right[0:4]}")

if compare_cpu:
    left_cpu = left.cpu()
    right_cpu = right.cpu()

    result_cpu = torch.zeros_like(left, device='cpu')
    start_time = time.time()
    for i in range(0, result_cpu.shape[0]):
        result_cpu[i] = mul_op(left_cpu[i], right_cpu[i])
    stop_time = time.time()
    print(f"Multiplying {size[0]} quaternions on CPU took {1000 * (stop_time - start_time)} ms")

    print(f"result_cpu[0:4] = {result_cpu[0:4]}")

result_gpu = torch.zeros_like(left, device='cuda')
start_time = time.time()
result_gpu = mul_no_loop(left, right, result_gpu)
stop_time = time.time()

print(f"result_gpu[0:4] = {result_gpu[0:4]}")

print(f"Multiplying {size[0]} quaternions on GPU took {1000 * (stop_time - start_time)} ms")

bench_result_no_loop = triton.testing.do_bench(
        lambda: mul_no_loop(left, right, result_gpu))

print(f"Median runtime via do_bench (no loop) = {bench_result_no_loop * 1000} ms")

bench_result_loop = triton.testing.do_bench(
        lambda: mul_loop(left, right, result_gpu))

print(f"Median runtime via do_bench (with loop) = {bench_result_loop * 1000} ms")

if compare_cpu:
    diff = result_gpu.cpu() - result_cpu
    error = torch.sum(torch.sqrt(diff * diff))
    avg_error = error / result_cpu.numel()
    print(f"Error is {error}, average error per element = {avg_error}.")

