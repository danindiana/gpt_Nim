To compile and run a Nim program that utilizes CUDA across both GPUs (NVIDIA GeForce RTX 3060 and NVIDIA GeForce GTX 1060 6GB), we need to ensure that our code can handle multi-GPU setups. Below is a guide on how to compile and run such a program.

### Step 1: Prepare CUDA Source Files

We'll create a CUDA source file (`cuda_multi_gpu.cu`) that can handle computations on multiple GPUs.

#### cuda_multi_gpu.cu

```c
#include <cuda_runtime.h>
#include <stdio.h>

__global__
void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}

void run_add(int n, float *x, float *y, int device_id) {
    cudaSetDevice(device_id);
    float *d_x, *d_y;

    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    add<<<(n + 255) / 256, 256>>>(n, d_x, d_y);

    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
}
```

### Step 2: Create CUDA Header File

#### cuda_multi_gpu.h

```c
#ifndef CUDA_MULTI_GPU_H
#define CUDA_MULTI_GPU_H

void run_add(int n, float *x, float *y, int device_id);

#endif
```

### Step 3: Compile CUDA Source File

```bash
nvcc -c -o cuda_multi_gpu.o cuda_multi_gpu.cu
```

### Step 4: Create Nim Binding File

#### cuda_multi_gpu.nim

```nim
{.passL: "cuda_multi_gpu.o", passL: "-lcudart".}

type
  float32* {.importc: "float", header: "cuda_multi_gpu.h".} = float
  PtrFloat* {.importc: "float*", header: "cuda_multi_gpu.h".} = ptr float

proc run_add*(n: int32, x: PtrFloat, y: PtrFloat, device_id: int32) {.importc, dynlib: "cuda_multi_gpu".}
```

### Step 5: Create Main Nim Program

#### main.nim

```nim
import strutils
import sequtils
import cuda_multi_gpu

proc checkCUDAError(errorMessage: cstring) =
  let err = cudaGetLastError()
  if err != cudaSuccess:
    echo "CUDA error: ", $err, " at ", errorMessage

when isMainModule:
  let N = 1_000_000
  var x: seq[float32] = newSeq[float32](N)
  var y: seq[float32] = newSeq[float32](N)

  for i in 0..<N:
    x[i] = 1.0'f32
    y[i] = 2.0'f32

  # Split the workload across two GPUs
  let halfN = N div 2

  # Create subarrays for each GPU
  var x1 = x[0..<halfN]
  var y1 = y[0..<halfN]
  var x2 = x[halfN..<N]
  var y2 = y[halfN..<N]

  # Run the CUDA addition on both GPUs
  run_add(halfN, cast[ptr float32](x1.unsafeAddr), cast[ptr float32](y1.unsafeAddr), 0)
  run_add(halfN, cast[ptr float32](x2.unsafeAddr), cast[ptr float32](y2.unsafeAddr), 1)

  # Combine results back into the original array
  for i in 0..<halfN:
    y[i] = y1[i]
    y[halfN + i] = y2[i]

  # Check for CUDA errors
  checkCUDAError("add")

  # Output sample results
  echo "Sample output y[0..4]: ", y[0..4]
```

### Step 6: Compile and Run the Nim Program

1. **Compile the CUDA source file**:

```bash
nvcc -c -o cuda_multi_gpu.o cuda_multi_gpu.cu
```

2. **Compile and run the Nim program**:

```bash
nim c -r main.nim
```

### Explanation

1. **CUDA Source Files**:
   - `cuda_multi_gpu.cu`: Contains the kernel function `add` and a function `run_add` to run the kernel on a specified device.
   - `cuda_multi_gpu.h`: Header file for the `cuda_multi_gpu.cu`.

2. **Nim Binding File**:
   - `cuda_multi_gpu.nim`: Defines the types and procedures for calling the CUDA functions from Nim.

3. **Main Nim Program**:
   - `main.nim`: Initializes the data, splits the workload across two GPUs, runs the CUDA kernel on both GPUs, and combines the results.

By following these steps, you can compile and run a Nim program that utilizes CUDA across both of your GPUs.
