#include<iostream>
#include<stdio.h>
#include<vector>
#include<cassert>
#include<chrono>
#include<cmath>
#include "cuda_profiler_api.h"
#include "matrix.h"
#include "ReadWriteData.h"
#include "header.h"
#include "PrecondBiCGSTAB.h"
#include "SolverResults.h"
#include "factorization.h"

#include "ILU_0.h"

namespace {

#include "Bicgstab_common.hpp"


__device__ __forceinline__ void legacy_sparse_lower_triangular_solve_mixed(const int num_rows, const int* const __restrict__ row_ptrs, const int* const __restrict__ col_inds, 
    const double* const __restrict__ factored_array_page, const int* const __restrict__ diag_info, const double* const __restrict__ vec_shared, volatile double* __restrict__ const temp_vec_shared)
{   
    
        const int row_index = threadIdx.x;

        if(row_index >= num_rows)
        {
            return;
        }

        double sum = 0;

        const int start = row_ptrs[row_index];
        const int end = diag_info[row_index];
        
        int i = start;
        
        
        bool completed = false;

        while(!completed)
        {   
            const int col_index = col_inds[i];

            if( i < end  &&  isfinite(temp_vec_shared[col_index]))
            {
                sum += factored_array_page[i] * temp_vec_shared[col_index];
                i++;
            }
            
            if(i == end)
            {   
                temp_vec_shared[row_index] = (vec_shared[row_index] - sum)/1; // 1 in place of factored_array_page[end]
                
                completed = true;
                
            }

            
        }

}


__device__ __forceinline__ void legacy_sparse_upper_triangular_solve_mixed(const int num_rows, const int* const __restrict__ row_ptrs, const int* const __restrict__ col_inds, 
    const double* const __restrict__ factored_array_page, const int* const __restrict__ diag_info, volatile const double* const __restrict__ temp_vec_shared, volatile double* const __restrict__ vec_hat_shared)
{
    const int row_index = threadIdx.x;

    if(row_index >= num_rows)
    {
        return;
    }

    double sum = 0;

    const int start = diag_info[row_index];
    const int end = row_ptrs[row_index + 1]  - 1;
    int i = end;

    bool completed = false;

    while(!completed )
    {   
       

        const int col_index = col_inds[i];

        if( i > start && isfinite(vec_hat_shared[col_index]))
        {
            sum += factored_array_page[i] * vec_hat_shared[col_index];
            i--;
        }

      
        if(i == start)
        {
            vec_hat_shared[row_index] = (temp_vec_shared[row_index] - sum)/factored_array_page[start];
           
            completed = true;
        }

      
    }

}



__device__ __forceinline__ void ApplyPreconditionerILU_mixed(const int num_rows, const int* const __restrict__ row_ptrs, const int* const __restrict__ col_inds, 
    const double* const __restrict__ factored_array_page, const int* const __restrict__ diag_info, const double* const __restrict__ vec_shared, volatile double* const __restrict__ vec_hat_shared, double* const temp_ilu_requirements_shared)
{
     // vec_hat = precond * vec
    // => L * U  * vec_hat = vec
    // => L * y = vec , find y , and then U * vec_hat = y, find vec_hat

    //sparse triangular solves

    //if we want to use the busy waiting while loop approach, then the num_rows should be <= threadblock size, else there is possibility of a deadlock!
    assert(num_rows <= blockDim.x);
    //TODO: For upper trsv, use thread 0 for the bottommost row, this way we could avoid :  assert(num_rows <= blockDim.x), as there won't be a possibility of deadlock then!

    // __shared__  volatile double temp_vec_shared[MAX_NUM_ROWS];
    volatile double* __restrict__ temp_vec_shared = temp_ilu_requirements_shared;

    for(int i = threadIdx.x ; i < num_rows; i += blockDim.x)
    {
        temp_vec_shared[i] = 1.8/0; //TODO: find a better way to deal with this!
        vec_hat_shared[i] = 1.3/0;

    }

    __syncthreads();
    
    
    legacy_sparse_lower_triangular_solve_mixed(num_rows, row_ptrs, col_inds, factored_array_page, diag_info, vec_shared, temp_vec_shared);

    __syncthreads();

    legacy_sparse_upper_triangular_solve_mixed(num_rows, row_ptrs, col_inds, factored_array_page, diag_info, temp_vec_shared, vec_hat_shared);
}



__global__ void KernelBatchedPreconditionedBiCGSTAB_ILU(const int num_rows, const int num_nz, const int num_pages, const int* const __restrict__ row_ptrs, 
    const int* const __restrict__ col_inds, const double* const __restrict__ vals_mat, const double* const __restrict__ vals_rhs, double* const __restrict__ vals_ans,
    double* const __restrict__ factored_array, const int* const __restrict__ diag_info,float* const __restrict__ iter_counts , int* const __restrict__ conv_flags, 
    double* const __restrict__ iter_residual_norms)
{
  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~shared memory ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

   extern __shared__ double shared_mem[];
   double* __restrict__ x_shared = shared_mem;
   double* __restrict__ r_shared = x_shared + num_rows;
   double* __restrict__ r_hat_shared = r_shared + num_rows;
   double* __restrict__ p_shared = r_hat_shared + num_rows;
   double* __restrict__ v_shared = p_shared + num_rows;
   double* __restrict__ s_shared = v_shared + num_rows;
   double* __restrict__ t_shared = s_shared + num_rows;
   double* __restrict__ s_hat_shared = t_shared + num_rows;
   double* __restrict__ p_hat_shared = s_hat_shared + num_rows;

   double* __restrict__ temp_ilu_requirements_shared = p_hat_shared + num_rows; // 1 * num_rows

   
   
   /*
    __shared__ double x_shared[MAX_NUM_ROWS];
    __shared__ double r_shared[MAX_NUM_ROWS];
    __shared__ double r_hat_shared[MAX_NUM_ROWS];
    __shared__ double p_shared[MAX_NUM_ROWS];
    __shared__ double v_shared[MAX_NUM_ROWS];
    __shared__ double s_shared[MAX_NUM_ROWS];
    __shared__ double t_shared[MAX_NUM_ROWS];
   // __shared__ double r_true_shared[MAX_NUM_ROWS];
    

    __shared__ double s_hat_shared[MAX_NUM_ROWS];
    __shared__ double p_hat_shared[MAX_NUM_ROWS];
 
    */

    int page_id = blockIdx.x;


    if(page_id < num_pages)
    {   


        
        /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ shared memory initialization/assigments~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
        initialization(num_rows, num_nz, row_ptrs, col_inds, vals_mat, vals_rhs, x_shared, v_shared, p_shared, r_shared, r_hat_shared);
        
        __syncthreads();

        /*---------------------------------------------- preconditioner already generated------------------------------------------*/


        __shared__ double res_initial;
        __shared__ double rho_old;
        __shared__ double rho_new;
        __shared__ double omega_old;
        __shared__ double omega_new;
        __shared__ double alpha;
        __shared__ double beta;
        __shared__ double iter_residual_norm;
        __shared__ double b_norm;
        __shared__ int conv_flag;

        L2Norm(num_rows, r_shared, res_initial); 

        __syncthreads();

        if(threadIdx.x == 0)
        {
            iter_residual_norm = res_initial;
            rho_old = 1;
            rho_new = 1;
            omega_old = 1;
            omega_new = 1;
            alpha = 1;
            beta = 1;
            conv_flag = -1;
        }
          
       
        L2Norm(num_rows, vals_rhs + page_id*num_rows, b_norm);

        __syncthreads();
        
        if(b_norm == 0)
        {   
            for(int i = threadIdx.x; i < num_rows ; i += blockDim.x)
                x_shared[i] = 0;


            if(threadIdx.x == 0 )
            {   
                printf(" RHS for problem id: %d is 0. x = 0 is the solution. ",page_id);

                iter_counts[page_id] = 0;
                conv_flags[page_id] = 1;
                iter_residual_norms[page_id] = 0;
            }    

            __syncthreads();
        
        }
        else
        {
            if(res_initial < ATOL )
            {   
                if(threadIdx.x == 0 )
                {   
                    printf("\n Initial guess for problem id: %d is good enough. No need of iterations. \n", page_id);


                    iter_counts[page_id] = 0;
                    conv_flags[page_id] = 1;
                    iter_residual_norms[page_id] = res_initial;
                }	    
            }
            else
            {
                 /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Then can start iterating ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
                    float iter = 0;
                
                    while(iter < MAX_ITER)
                    {
                        inner_product(num_rows, r_shared, r_hat_shared, rho_new);
                        __syncthreads();

                        if(rho_new == 0)
                        {
                            if(threadIdx.x == 0)
                            {
                                printf("\n Method failed for problem id: %d\n",page_id);
                            }

                            break;
                        }
                        
                        if(threadIdx.x == 0)
                        {
                            beta = (rho_new/ rho_old) * ( alpha/ omega_old);
                        }
                        __syncthreads();
                    
                        
                        Update_p(num_rows, p_shared, r_shared ,v_shared, beta, omega_old);
                        __syncthreads();
                        

                        ApplyPreconditionerILU_mixed(num_rows, row_ptrs, col_inds, factored_array + page_id * num_nz, diag_info , p_shared, p_hat_shared, temp_ilu_requirements_shared);

                        __syncthreads();

                        SpMV(num_rows, row_ptrs,col_inds, vals_mat + page_id*num_nz, p_hat_shared, v_shared);
                        __syncthreads(); 
                        
                        
                        __shared__ double r_hat_and_v_inner_prod;
                        inner_product(num_rows,r_hat_shared,v_shared, r_hat_and_v_inner_prod);
                        __syncthreads();


                        if(threadIdx.x == 0)
                        {
                            alpha = rho_new/r_hat_and_v_inner_prod;   
                        }
                        __syncthreads();  
                    

                        Update_s(num_rows,s_shared,r_shared,alpha,v_shared);
                        __syncthreads();
                        

                        L2Norm(num_rows, s_shared, iter_residual_norm); //an estimate
                        __syncthreads();

                        iter = iter + 0.5;

                      
                        if( iter_residual_norm < ATOL)
                        {
                            Update_x_middle(num_rows,x_shared,p_hat_shared,alpha);
                            __syncthreads();

                            if(threadIdx.x == 0)
                            {
                                conv_flag = 1;
                            }
                            __syncthreads();
                         
                            break;
    
                        }

                     
                        ApplyPreconditionerILU_mixed(num_rows, row_ptrs, col_inds, factored_array + page_id * num_nz, diag_info , s_shared, s_hat_shared, temp_ilu_requirements_shared);
                        __syncthreads();


                        SpMV( num_rows, row_ptrs , col_inds, vals_mat + page_id*num_nz , s_hat_shared, t_shared);
                        __syncthreads();
                    


                        __shared__ double t_and_s_inner_prod;
                        inner_product(num_rows,t_shared,s_shared, t_and_s_inner_prod);
                        __shared__ double t_and_t_inner_prod;
                        inner_product(num_rows,t_shared,t_shared, t_and_t_inner_prod);
                        __syncthreads();

                        if(threadIdx.x == 0)
                        {
                            omega_new = t_and_s_inner_prod/t_and_t_inner_prod;
                        }
                       __syncthreads();

                        Update_x(num_rows,x_shared,p_hat_shared,s_hat_shared,alpha,omega_new);
                        __syncthreads();
                        
                        
                        iter = iter + 0.5;


                        Update_r(num_rows,r_shared,s_shared,t_shared,omega_new);
                        __syncthreads();

                        L2Norm(num_rows,r_shared, iter_residual_norm);
                        __syncthreads();

                        if(threadIdx.x == 0)
                        {
                            rho_old = rho_new;
                            omega_old = omega_new;
                        }
                        __syncthreads();

                        
                        if( iter_residual_norm < ATOL)
                        {   
                            if(threadIdx.x == 0)
                            {
                                conv_flag = 1;
                            }
                            __syncthreads();
                          
                            break;
                        }

                        
                    }

                    __syncthreads();

                   /*  ComputeResidualVec(num_rows, row_ptrs , col_inds, vals_mat + page_id*num_nz, vals_rhs + page_id*num_rows, x_shared,r_true_shared);
                    __syncthreads();
                
                    
                    double true_resi_norm = L2Norm(num_rows,r_true_shared); */
                    
                    if(threadIdx.x == 0 )
                    {   
                      // printf("\nConv flag for problem_id: %d is %d , iter resi norm : %0.17lg, true resi norm: %0.17lg, iter:%f ",page_id,conv_flag, iter_residual_norm, true_resi_norm, iter );
                        iter_counts[page_id] = iter;
                        conv_flags[page_id] = conv_flag;
                        iter_residual_norms[page_id] = iter_residual_norm;
                    }

            }

        }

       
       // At the end,copy x_shared to global memory.
        for(int i = threadIdx.x; i < num_rows; i += blockDim.x)
            vals_ans[i + page_id*num_rows] = x_shared[i];

    
    }

}


int Batched_BiCGSTAB_ILU_Gpu_helper(const PagedCSRMatrices & A_pages,const PagedVectors& b_pages,PagedVectors & x_pages, 
    SolverResults & solver_results )
{
    std::cout << "\n\n-------------------------------------------------------------------------------\n Batched_Preconditioned BiCGSTAB_Gpu_helper " << std::endl;
    
   
    auto start = std::chrono::high_resolution_clock::now();
    
    //ilu preconditioner prep. phase- find diag ptrs 
    
    // (All these matrices are already sorted.(sorted while storing))
    int* diag_info = nullptr;
    cudaMalloc((void**)&diag_info, sizeof(int) * A_pages.GetNumRows());

    int num_missing_diagonal_eles = Count_Missing_Diagonal_Elements(A_pages , diag_info);

    if(num_missing_diagonal_eles > 0)
    {
       assert(0);
    }

    Find_locations_of_diagonal_elements(A_pages, diag_info);
    

    PagedCSRMatrices Factored_Pages;
    Copy_Gpu_PagedCSRMatrices(A_pages , Factored_Pages);

    ComputeILU0Approach1(Factored_Pages , diag_info);

    
    dim3 block(THREADS_PER_BLOCK,1,1);
    dim3 grid_solver(A_pages.GetNumPages(),1,1 );

    //------------------------------------------------------------------------------- Call main solver kernel-------------------------------------------------//

    const int dynamic_shared_mem_bytes =  10 * A_pages.GetNumRows() * sizeof(double);

    KernelBatchedPreconditionedBiCGSTAB_ILU<<< grid_solver, block , dynamic_shared_mem_bytes  >>>(A_pages.GetNumRows(), A_pages.GetNumNz(), A_pages.GetNumPages(),
    A_pages.GetPtrToGpuRowPtrs(),A_pages.GetPtrToGpuColInd(), A_pages.GetPtrToGpuValues(), b_pages.GetPtrToGpuValues(), x_pages.GetPtrToGpuValues(),
    Factored_Pages.GetPtrToGpuValues(), diag_info , solver_results.GetPtrToGpuIterCount(), solver_results.GetPtrToGpuConvFlag() , solver_results.GetPtrToGpuIterResNorm());

    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    cudaFree(diag_info);
    
    
    std::cout << " Time taken is: "  << (double)duration.count() << " microseconds\n\n ";  

    solver_results.SetTimeTaken((double)duration.count()/ 1000);


    //fill it with true residual norms
    KernelFillTrueResNorms<<< grid_solver , block , 0  >>>(A_pages.GetNumRows(), A_pages.GetNumNz(), A_pages.GetNumPages(), A_pages.GetPtrToGpuRowPtrs(),
    A_pages.GetPtrToGpuColInd(), A_pages.GetPtrToGpuValues(), b_pages.GetPtrToGpuValues(), x_pages.GetPtrToGpuValues(), solver_results.GetPtrToGpuTrueResNorm());

    cudaDeviceSynchronize();

    return 1;
}



} //unnamed namespace


//----------------------------------------------------------------------------------------------------------------------------------------------------------------


// A*x = b
void Batched_conv_ILU_app1_Preconditioned_BiCGSTAB_segregated_Gpu(const std::vector<std::string> & subdir, const PagedCSRMatrices & A_pages,
    const PagedVectors& b_pages,PagedVectors & x_pages,const bool is_scaled,  SolverResults & solver_results  )
{   

    assert(A_pages.ExistsGPU() == true);
    assert(b_pages.ExistsGPU() == true);
    assert(x_pages.ExistsGPU() == true);

    const int num_pages = A_pages.GetNumPages();
    assert(num_pages == b_pages.GetNumPages());
    assert(num_pages == x_pages.GetNumPages());

    const int num_rows = A_pages.GetNumRows();
    const int num_cols = A_pages.GetNumCols();
    
    assert(num_rows == num_cols);
    assert(num_cols == x_pages.GetNumElements());
    assert(num_rows == b_pages.GetNumElements());

   
    int success_code = 0;

    success_code = Batched_BiCGSTAB_ILU_Gpu_helper(A_pages,b_pages,x_pages, solver_results);

    std::string solution_file;

    if(is_scaled == true)
        solution_file = "x_scaled_gpu_conv_ilu_segregated_app1_bicgstab.mtx";
    else
        solution_file = "x_gpu_conv_ilu_segregated_app1_bicgstab.mtx";

    if(success_code == 1)
    {
        x_pages.CopyFromGpuToCpu();
        Print_ans(subdir,x_pages, solution_file);
        std::cout << "files containing soluation: x  are produced...  ( " <<  solution_file <<  " ) in their respective directories " << std::endl;

    }


}
