

//TODO: parallel reductions(norm, inner product)

__device__ __forceinline__ void ComputeResidualVec(const int num_rows,const int* const __restrict__ A_row_ptrs_shared,const int* const __restrict__ A_col_inds_shared,
    const double* const __restrict__ A_vals_shared,const double* const __restrict__ b_shared,const double* const __restrict__ x_shared, double* const __restrict__ res_shared)
{
    
    int num_warps_in_block = blockDim.x/WARP_SIZE;
    int lane = threadIdx.x & (WARP_SIZE -1);
    int local_warp_index = threadIdx.x/WARP_SIZE; //local warp index in a block

    for(int i = local_warp_index; i < num_rows ; i = i + num_warps_in_block)
    {
        int start_ind_for_row = A_row_ptrs_shared[i];
        int end_ind_for_row = A_row_ptrs_shared[i + 1];

        double temp = 0;

        for(int k = start_ind_for_row + lane; k < end_ind_for_row; k = k + WARP_SIZE)
        {
            temp += A_vals_shared[k]*x_shared[A_col_inds_shared[k]];
        }

        double val = temp;

        //warp level reduction
        for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    
        if(lane == 0)
        {   
            res_shared[i] = b_shared[i] - val;
        }

       

    }

}



__device__ void block_reduce(double* data)
{
    int nt = blockDim.x;
    int tid = threadIdx.x;

    for (int k = nt / 2; k > 0; k = k / 2)
    {
        __syncthreads();
        if (tid < k)
        {
            data[tid] += data[tid + k];
        }
    }


}

__device__ double inner_product1(const int num_rows, const double* const vec1_shared, const double* const vec2_shared, double* const temp_shared)
{   
    double tmp = 0;

    for(int i = threadIdx.x; i < num_rows; i = i + blockDim.x)
    {
        tmp = tmp + vec1_shared[i]*vec2_shared[i];
    }

    temp_shared[threadIdx.x] = tmp;

    __syncthreads();

    block_reduce(temp_shared);

    __syncthreads();

    return temp_shared[0];

}

__device__ double inner_product(const int num_rows, const double* const vec1_shared, const double* const vec2_shared)
{   
    double tmp = 0;

    for(int i=0; i < num_rows; i++)
        tmp = tmp + vec1_shared[i]*vec2_shared[i];
    

    __syncthreads();

   
   return tmp;

}

__device__ __forceinline__ void inner_product(const int num_rows, const double* const vec1_shared, const double* const vec2_shared, double & result_shared)
{   
   
    const int local_warp_id = threadIdx.x/WARP_SIZE;
    const int id_in_warp = threadIdx.x%WARP_SIZE;

    //use first warp
    if(local_warp_id == 0)
    {   
        double temp = 0;

        for(int idx = id_in_warp; idx < num_rows; idx += WARP_SIZE)
        {
            temp += vec1_shared[idx] * vec2_shared[idx];
        }

        double val = temp;

        for(int offset = WARP_SIZE/2; offset > 0; offset /= 2)
        {
              val += __shfl_down_sync(FULL_MASK, val, offset);
        }

        if(id_in_warp == 0)
        {
            result_shared = val; 
        }

    }


}



__device__ double L2Norm(const int num_rows,const double* const vec_shared)
{
    return sqrt(inner_product(num_rows,vec_shared,vec_shared));
}


__device__ __forceinline__ void L2Norm(const int num_rows,const double* const __restrict__ vec_shared, double & result_shared)
{
    inner_product(num_rows,vec_shared,vec_shared, result_shared);
    
    if(threadIdx.x == 0)
    {
        result_shared = sqrt(result_shared); 
    }

}



__device__ __forceinline__ void initialization(const int num_rows, const int num_nz,const int* const __restrict__ row_ptrs,const int* const __restrict__ col_inds,
    const double* const __restrict__ vals_mat,const double* const __restrict__ vals_rhs ,double* const __restrict__ x_shared,double* const __restrict__ v_shared,double* const __restrict__ p_shared,
double* const __restrict__ r_shared,double* const __restrict__ r_hat_shared)
{
    int num_warps_in_block = blockDim.x/WARP_SIZE;
    int local_thread_id = threadIdx.x; //local thread id in block
    int local_warp_index = threadIdx.x/WARP_SIZE; //local warp index in a block
    int page_id = blockIdx.x;
    int lane  = threadIdx.x & (WARP_SIZE -1);

    
    // x:initialize with 0s {Later on, have a provision for user's choice. So, may be x_pages: initialize--> with something n copy that to here}
    // r = b - A*x
    // r_hat = r
    // rho, alpha, omega
    // v with 0s
    // p with 0s

   

    for(int i = local_thread_id ; i < num_rows; i = i + blockDim.x)
    {   
        x_shared[i] = 0.00;
        v_shared[i] = 0.00;
        p_shared[i] = 0.00;
        
    }

    __syncthreads();

    //initialize r
    ComputeResidualVec(num_rows, row_ptrs, col_inds, vals_mat + page_id*num_nz, vals_rhs + page_id*num_rows, x_shared,r_shared);
    __syncthreads();

    
    for(int i = local_warp_index*WARP_SIZE  + lane ; i < num_rows ; i = i + num_warps_in_block*WARP_SIZE)
    {   
        r_hat_shared[i] = r_shared[i];
    }

    
}


__device__ __forceinline__ void Update_p(const int num_rows,double* const __restrict__ p_shared,const double* const __restrict__ r_shared,const double* const __restrict__ v_shared,
    const double & beta,const double & omega_old)
{
    
    for(int i = threadIdx.x ; i < num_rows; i = i + blockDim.x)
    {   
        double val = r_shared[i] + beta*(p_shared[i] - omega_old*v_shared[i]);
        p_shared[i] = val;
        
    }

} 


__device__ __forceinline__ void Update_s(const int num_rows,double* const __restrict__ s_shared,const double* const __restrict__ r_shared,
const double & alpha,const double* const __restrict__ v_shared)
{
    for(int i = threadIdx.x; i < num_rows; i = i + blockDim.x)
    {
        s_shared[i] = r_shared[i] - alpha*v_shared[i];
    }
}

__device__ __forceinline__ void Update_x(const int num_rows,double* const __restrict__ x_shared,const double* const __restrict__ p_shared,
const double* const __restrict__ s_shared, const double & alpha, const double & omega_new)
{
    for(int i = threadIdx.x; i < num_rows; i = i + blockDim.x)
    {
        x_shared[i] = x_shared[i] + alpha*p_shared[i] + omega_new*s_shared[i];
    }
}


__device__ __forceinline__ void Update_x_middle(const int num_rows, double* const __restrict__ x_shared,const double* const __restrict__ p_shared, const double & alpha)
{   
    for(int i = threadIdx.x; i < num_rows; i = i + blockDim.x)
    {
        x_shared[i] = x_shared[i] + alpha*p_shared[i] ;
    }

}



__device__ __forceinline__ void Update_r(const int num_rows,double* const __restrict__ r_shared,const double* const __restrict__ s_shared,const double* const __restrict__ t_shared,const double & omega_new)
{
    
    for(int i = threadIdx.x; i < num_rows; i = i + blockDim.x)
    {
        r_shared[i] = s_shared[i] - omega_new*t_shared[i];
    }

    
}





__device__ __forceinline__ void SpMV(const int num_rows,const int* const __restrict__ mat_row_ptrs_shared,const int* const __restrict__ mat_col_inds_shared,
    const double* const __restrict__ mat_vals_shared,const double* const __restrict__ vec_shared,double* const __restrict__ ans_shared)
{
  
    int num_warps_in_block = blockDim.x/WARP_SIZE;
    int lane = threadIdx.x & (WARP_SIZE -1);
    int local_warp_index = threadIdx.x/WARP_SIZE; //local warp index in a block

    for(int i = local_warp_index; i < num_rows ; i = i + num_warps_in_block)
    {
        int start_ind_for_row = mat_row_ptrs_shared[i];
        int end_ind_for_row = mat_row_ptrs_shared[i + 1];

        double temp = 0;

        for(int k = start_ind_for_row + lane; k < end_ind_for_row; k = k + WARP_SIZE)
        {
            temp += mat_vals_shared[k]*vec_shared[mat_col_inds_shared[k]];
        }

        double val = temp;

        //warp level reduction
        for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    

        if(lane == 0)
        {   
            ans_shared[i] = val;
        }

        

    }

}



__global__ void KernelFillTrueResNorms(const int num_rows, const int num_nz, const int num_pages, const int* const row_ptrs, 
    const int* const col_inds, const double* const vals_mat, const double* const vals_rhs, const double* const vals_ans, double* const true_residual_norms)
{
    __shared__ int A_row_ptrs_shared[MAX_NUM_ROWS + 1];
    __shared__ int A_col_inds_shared[MAX_NUM_NZ];
    __shared__ double A_vals_shared[MAX_NUM_NZ];
    __shared__ double b_shared[MAX_NUM_ROWS];
    __shared__ double x_shared[MAX_NUM_ROWS];

    __shared__ double r_true_shared[MAX_NUM_ROWS];

    int page_id = blockIdx.x;
    
    if(page_id < num_pages)
    {

            for(int i = threadIdx.x; i < num_rows + 1; i = i + blockDim.x)
            {   
                A_row_ptrs_shared[i] = row_ptrs[i];

            }


            for(int i = threadIdx.x ; i < num_nz; i = i + blockDim.x)
            {   
                A_col_inds_shared[i] = col_inds[i];
                A_vals_shared[i] = vals_mat[i + page_id*num_nz];

            }

            for(int i = threadIdx.x ; i < num_rows; i = i + blockDim.x)
            {   
                b_shared[i] = vals_rhs[i + page_id*num_rows];
                x_shared[i] = vals_ans[i + page_id*num_rows];   
            }

            __syncthreads();

            ComputeResidualVec(num_rows, A_row_ptrs_shared, A_col_inds_shared, A_vals_shared, b_shared, x_shared,r_true_shared);
            __syncthreads();


            double true_resi_norm = L2Norm(num_rows,r_true_shared);

            if(threadIdx.x == 0)
                true_residual_norms[page_id] = true_resi_norm;
    }

}    
