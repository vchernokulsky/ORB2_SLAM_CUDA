#include "../IC_Angles.h"
#include "IC_Angles.cuh"


static  __global__ void
IC_Angle_kernel(const cv::cuda::PtrStepb image, vx_keypoint_t *kp_buf, vx_size kp_size, vx_size kp_stride,
                vx_int32 *u_max_buf,
                vx_size u_max_size, vx_size u_max_stride) {
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


        for (u = threadIdx.x - u_max_size; u <= u_max_size; u += blockDim.x)
            m_10 += u * image(loc.y, loc.x + u);

        cv::cuda::device::reduce<32>(srow0, m_10, threadIdx.x, op);


        int v_sum;
        int m_sum;
        int d;
        int val_plus;
        int val_minus;

        for (int v = 1; v <= u_max_size; ++v) {
            v_sum = 0;
            m_sum = 0;
            d = vxArrayItem(vx_int32, u_max_buf, v, u_max_stride);

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
            float kp_dir = atan2f((float) m_01, (float) m_10);
            kp_dir += (kp_dir < 0) * (2.0f * CV_PI_F);
            kp_dir *= 180.0f / CV_PI_F;

            vxArrayItem(vx_keypoint_t, kp_buf, ptidx, kp_stride).orientation = kp_dir;
        }
    }
}

void IC_Angles_gpu(const cv::cuda::GpuMat &image, vx_keypoint_t *kp_buf, vx_size kp_size, vx_size kp_stride,
                   vx_int32 *u_max_buf,
                   vx_size u_max_size, vx_size u_max_stride, cudaStream_t stream) {
    dim3 block(32, 8);

    dim3 grid(cv::cuda::device::divUp(kp_size, block.y));

    IC_Angle_kernel<<<grid, block, 0, stream>>>(image, kp_buf, kp_size, kp_stride, u_max_buf, u_max_size, u_max_stride);

    if (stream == 0)
        cudaSafeCall(cudaDeviceSynchronize());
}
