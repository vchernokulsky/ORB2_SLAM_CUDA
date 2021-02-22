#include "../IC_Angles.h"
#include "IC_Angles.cuh"

__constant__ int c_u_max[32];

void loadUMax(const int* u_max, int count)
{
    checkCudaErrors(( cudaMemcpyToSymbol(c_u_max, u_max, count * sizeof(int))));
}

static  __global__ void IC_Angle_kernel(const cv::cuda::PtrStepb image, cv::KeyPoint * keypoints, const int npoints)
{
    __shared__ int smem0[8 * 32];
    __shared__ int smem1[8 * 32];

    int* srow0 = smem0 + threadIdx.y * blockDim.x;
    int* srow1 = smem1 + threadIdx.y * blockDim.x;

    cv::cuda::device::plus<int> op;

    const int ptidx = blockIdx.x * blockDim.y + threadIdx.y;

    if (ptidx < npoints)
    {
        int m_01 = 0, m_10 = 0;

        const short2 loc = make_short2(keypoints[ptidx].pt.x, keypoints[ptidx].pt.y);

        for (int u = threadIdx.x - HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; u += blockDim.x)
            m_10 += u * image(loc.y, loc.x + u);

        cv::cuda::device::reduce<32>(srow0, m_10, threadIdx.x, op);

        for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
        {
            int v_sum = 0;
            int m_sum = 0;
            const int d = c_u_max[v];

            for (int u = threadIdx.x - d; u <= d; u += blockDim.x)
            {
                int val_plus = image(loc.y + v, loc.x + u);
                int val_minus = image(loc.y - v, loc.x + u);

                v_sum += (val_plus - val_minus);
                m_sum += u * (val_plus + val_minus);
            }

            cv::cuda::device::reduce<32>(cv::cuda::device::smem_tuple(srow0, srow1), thrust::tie(v_sum, m_sum), threadIdx.x, thrust::make_tuple(op, op));

            m_10 += m_sum;
            m_01 += v * v_sum;
        }

        if (threadIdx.x == 0)
        {
            float kp_dir = atan2f((float)m_01, (float)m_10);
            kp_dir += (kp_dir < 0) * (2.0f * CV_PI_F);
            kp_dir *= 180.0f / CV_PI_F;

            keypoints[ptidx].angle = kp_dir;
        }
    }
}

void IC_Angle_gpu(cv::cuda::GpuMat &image, cv::KeyPoint * keypoints, int npoints, cudaStream_t stream)
{
    dim3 block(32, 8);

    dim3 grid;
    grid.x = cv::cuda::device::divUp(npoints, block.y);

    IC_Angle_kernel<<<grid, block, 0, stream>>>(image, keypoints, npoints);

    cudaSafeCall( cudaGetLastError() );

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}
