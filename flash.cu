#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void forward_kernel(
        const float* Q, const float* K, const float* V, 
        const int N, const int d, const int Tc, const int Tr, 
        const int Bc, const int Br, const float softmax_scale,
        float* l, float *m, float* O
) {
    // this is me changing the source code!
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}


// Note that the tensor inputs Q, K, V to this function are already expected be be on the GPU!
torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // TODO: determine Bc, Br dynamically
    const int Bc = 32; const int Br = 32;

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q); // created by default on the same device as Q (on the GPU)
    
    // auto l = torch::zeros({B, nh, N}); // created on the CPU by default.. need to send it to the GPU later
    // auto m = torch::full({B, nh, N}, -INFINITY); // crated on the CPU by default
    // torch::Device device(torch::kCUDA);
    // l = l.to(device); m = m.to(device); // send l, m to the GPU
    // // if the cuda kernel recevies a CPU pointer, it will crash! / undefined behaviour!

    // safer option is to make l, m inherit properties from Q (like datatype, device, etc):
    auto l = torch::zeros({B, nh, N}, Q.options());
    auto m = torch::full({B, nh, N}, -INFINITY, Q.options());
    

    // Calculate SRAM size needed per block
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n ", max_sram_size, sram_size);

    
    // computation of each block is independent of each other (head-wise independence!)
    // each block computes the results for (N*d) shaped {Q, K, V} matrices 
    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);  // Bc threads per block

    // need to get the raw C pointer from the Q torch::Tensor class!
    // and then pass the raw C pointer as an input to the cuda kernel
    float* q_ptr = Q.data_ptr<float>(); // these are pointers to GPU memory!
    float* k_ptr = K.data_ptr<float>();
    float* v_ptr = V.data_ptr<float>();
    float* l_ptr = l.data_ptr<float>();
    float* m_ptr = m.data_ptr<float>();
    float* o_ptr = O.data_ptr<float>();
    
    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        q_ptr, k_ptr, v_ptr,
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l_ptr, m_ptr, o_ptr
    );
    
    return O;
}