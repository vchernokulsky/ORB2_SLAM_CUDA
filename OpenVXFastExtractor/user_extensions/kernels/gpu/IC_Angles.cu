#include "../IC_Angles.h"
#include "IC_Angles.cuh"


__constant__ int c_u_max[32] = {15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

static  __global__ void
IC_Angle_kernel(const cv::cuda::PtrStepb image, vx_keypoint_t *kp_buf, vx_size kp_size, vx_size kp_stride) {
    __shared__ int smem0[8 * 32];
    __shared__ int smem1[8 * 32];

    int *srow0 = smem0 + threadIdx.y * blockDim.x;
    int *srow1 = smem1 + threadIdx.y * blockDim.x;

    cv::cuda::device::plus<int> op;

    const int ptidx = blockIdx.x * blockDim.y + threadIdx.y;

    if (ptidx < kp_size) {
        int m_01 = 0, m_10 = 0, u;

        const short2 loc = make_short2(vxArrayItem(vx_keypoint_t, kp_buf, ptidx, kp_stride).x,
                                       vxArrayItem(vx_keypoint_t, kp_buf, ptidx, kp_stride).y);


        for (u = threadIdx.x - HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; u += blockDim.x)
            m_10 += u * image(loc.y, loc.x + u);

        cv::cuda::device::reduce<32>(srow0, m_10, threadIdx.x, op);


        int v_sum;
        int m_sum;
        int d;
        int val_plus;
        int val_minus;

        for (int v = 1; v <= HALF_PATCH_SIZE; ++v) {
            v_sum = 0;
            m_sum = 0;
            d = c_u_max[v];

            for (u = threadIdx.x - d; u <= d; u += blockDim.x) {
                val_plus = image(loc.y + v, loc.x + u);
                val_minus = image(loc.y - v, loc.x + u);

                v_sum += (val_plus - val_minus);
                m_sum += u * (val_plus + val_minus);
            }

            cv::cuda::device::reduce<32>(cv::cuda::device::smem_tuple(srow0, srow1), thrust::tie(v_sum, m_sum),
                                         threadIdx.x, thrust::make_tuple(op, op));

            m_10 += m_sum;
            m_01 += v * v_sum;
        }

        if (threadIdx.x == 0) {
            vxArrayItem(vx_keypoint_t, kp_buf, ptidx, kp_stride).orientation = atan2f((float) m_01, (float) m_10);
            vxArrayItem(vx_keypoint_t, kp_buf, ptidx, kp_stride).orientation += (vxArrayItem(vx_keypoint_t, kp_buf, ptidx, kp_stride).orientation < 0) * (2.0f * CV_PI_F);
            vxArrayItem(vx_keypoint_t, kp_buf, ptidx, kp_stride).orientation *= 180.0f / CV_PI_F;
        }
    }
}

void IC_Angles_gpu(const cv::cuda::GpuMat &image, vx_keypoint_t *kp_buf, vx_size kp_size, vx_size kp_stride,
                   cudaStream_t stream) {
    dim3 block(32, 8);

    dim3 grid(cv::cuda::device::divUp(kp_size, block.y));

    IC_Angle_kernel<<<grid, block, 0, stream>>>(image, kp_buf, kp_size, kp_stride);
}