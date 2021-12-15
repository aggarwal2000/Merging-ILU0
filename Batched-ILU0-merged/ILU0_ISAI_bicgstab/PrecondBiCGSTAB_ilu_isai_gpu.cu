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

#include "parILU_0.h"
#include "ILU_0.h"
#include "isai.h"




namespace {

#include "Bicgstab_common.hpp"

__device__ void ApplyPreconditionerILU_ISAI(const int num_rows , const int* const aiL_row_ptrs, 
    const int* const aiL_col_idxs , const double* const aiL_values,  const int* const aiU_row_ptrs,
    const int* const aiU_col_idxs, const double* const aiU_values,  const double* const vec_shared, double* const vec_hat_shared)
{

    // vec_hat = precond * vec
    // => L * U  * vec_hat = vec
    // => aiL * L * U * vec_hat =  aiL * vec
    // => U * vec_hat = aiL * vec
    // => aiU * U * vec_hat = aiU * aiL * vec
    // => vec_hat = aiU * aiL * vec
  
    __shared__ double temp_vec_shared[MAX_NUM_ROWS];

    SpMV(num_rows, aiL_row_ptrs, aiL_col_idxs, aiL_values, vec_shared, temp_vec_shared);

    __syncthreads();

    SpMV(num_rows, aiU_row_ptrs, aiU_col_idxs, aiU_values, temp_vec_shared, vec_hat_shared);

}



__global__ void KernelBatchedPreconditionedBiCGSTAB_ILU_ISAI(const int num_rows, const int num_nz, const int num_pages, const int* const row_ptrs, 
    const int* const col_inds, const double* const vals_mat, const double* const vals_rhs, double* const vals_ans,
    const int aiL_nnz , const int* const aiL_row_ptrs, const int* const aiL_col_idxs, const double* const aiL_vals ,
    const int aiU_nnz, const int* const aiU_row_ptrs, const int* const aiU_col_idxs, const double* const aiU_vals,
    float* const iter_counts , int* const conv_flags, double* const iter_residual_norms)
{
  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~shared memory ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

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
 


    int page_id = blockIdx.x;


    if(page_id < num_pages)
    {   


        
        /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ shared memory initialization/assigments~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
        initialization(num_rows, num_nz, row_ptrs, col_inds, vals_mat, vals_rhs, x_shared, v_shared, p_shared, r_shared, r_hat_shared);
        
        __syncthreads();


        /*--------------------------------------------------- Preconditioner already generated ----------------------------------------------------*/


        double res_initial = L2Norm(num_rows, r_shared); 
          
        double iter_residual_norm = res_initial;

        double rho_old = 1;
        double rho_new = 1;
        double omega_old = 1;
        double omega_new = 1;
        double alpha = 1;
        double beta = 1; 

        double b_norm = L2Norm(num_rows, vals_rhs + page_id*num_rows);
        
        int conv_flag = -1;

        
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
                        rho_new = inner_product(num_rows, r_shared, r_hat_shared);

                        if(rho_new == 0)
                        {
                            if(threadIdx.x == 0)
                            {
                                printf("\n Method failed for problem id: %d\n",page_id);
                            }

                            break;
                        }
                        
                        beta = (rho_new/rho_old)*(alpha/omega_old);
                    
                        
                        Update_p(num_rows,p_shared,r_shared ,v_shared,beta,omega_old);
                        __syncthreads();
                        

                        ApplyPreconditionerILU_ISAI(num_rows, aiL_row_ptrs, aiL_col_idxs, aiL_vals + page_id * aiL_nnz, aiU_row_ptrs, aiU_col_idxs , aiU_vals + page_id * aiU_nnz, p_shared, p_hat_shared);

                        __syncthreads();

                        SpMV(num_rows, row_ptrs,col_inds, vals_mat + page_id*num_nz, p_hat_shared, v_shared);
                        __syncthreads(); 
                        
                        
                        double r_hat_and_v_inner_prod = inner_product(num_rows,r_hat_shared,v_shared);
                        alpha = rho_new/r_hat_and_v_inner_prod;        
                    

                        Update_s(num_rows,s_shared,r_shared,alpha,v_shared);
                        __syncthreads();
                        

                        iter_residual_norm = L2Norm(num_rows, s_shared); //an estimate
                        
                        iter = iter + 0.5;

                        if( iter_residual_norm < ATOL)
                        {
                            Update_x_middle(num_rows,x_shared,p_hat_shared,alpha);
                            __syncthreads();

                            conv_flag = 1;

                            
                            break;
    
                        }

                     
                        ApplyPreconditionerILU_ISAI(num_rows, aiL_row_ptrs, aiL_col_idxs, aiL_vals + page_id * aiL_nnz, aiU_row_ptrs, aiU_col_idxs , aiU_vals + page_id * aiU_nnz, s_shared, s_hat_shared);
                        __syncthreads();


                        SpMV( num_rows, row_ptrs , col_inds, vals_mat + page_id*num_nz , s_hat_shared, t_shared);
                        __syncthreads();
                    


                        double t_and_s_inner_prod = inner_product(num_rows,t_shared,s_shared);
                        double t_and_t_inner_prod = inner_product(num_rows,t_shared,t_shared);
                        omega_new = t_and_s_inner_prod/t_and_t_inner_prod;
                        

                        Update_x(num_rows,x_shared,p_hat_shared,s_hat_shared,alpha,omega_new);
                        __syncthreads();
                        
                        
                        iter = iter + 0.5;


                        Update_r(num_rows,r_shared,s_shared,t_shared,omega_new);
                        __syncthreads();

                        iter_residual_norm = L2Norm(num_rows,r_shared);
                        rho_old = rho_new;
                        omega_old = omega_new;

                        
                        if( iter_residual_norm < ATOL)
                        {   
                            conv_flag = 1;
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




int Batched_BiCGSTAB_ILU_ISAI_Gpu_helper(const PagedCSRMatrices & A_pages,const PagedVectors& b_pages,PagedVectors & x_pages, SolverResults & solver_results,
    const bool is_parilu , const int num_iter_par_ilu, double & PST, double & IST, double & OET )
{
    std::cout << "\n\n-------------------------------------------------------------------------------\n Batched_Preconditioned BiCGSTAB_Gpu_helper " << std::endl;
    
   
    auto start = std::chrono::high_resolution_clock::now();
    
    //generate ILU preconditioner
    PagedCSRMatrices L_pages;
    PagedCSRMatrices U_pages;


    if(is_parilu)
    {	
    	//std::cout << " \npar ilu with num iter: " << num_iter_par_ilu << std::endl; 
        ParILU_0_Factorization_Gpu(A_pages , L_pages, U_pages, num_iter_par_ilu);
    }
    else
    {
        const int approach_num = 3;
        //Note: For pele matrices, approach 1 works better as compared to the depenedency graph approach as the matrices are not that sparse. For other cases, approach 3 is exepected to be faster than others.
        //std::cout << " \nilu " << std::endl;
        ILU_0_Factorization_Gpu(A_pages , L_pages, U_pages, approach_num);
    }

    
    PagedCSRMatrices  aiL_pages;
    PagedCSRMatrices  aiU_pages;
    
    GenerateISAI_gpu(aiL_pages, L_pages, mat_type::lower_tri,2);	
  
    GenerateISAI_gpu(aiU_pages, U_pages, mat_type::upper_tri,2);	
 
    
   cudaDeviceSynchronize();
   auto mid = std::chrono::high_resolution_clock::now();

    dim3 block(THREADS_PER_BLOCK,1,1);
    dim3 grid_solver(A_pages.GetNumPages(),1,1 );

    //------------------------------------------------------------------------------- Call main solver kernel-------------------------------------------------//

    KernelBatchedPreconditionedBiCGSTAB_ILU_ISAI<<< grid_solver, block , 0  >>>(A_pages.GetNumRows(), A_pages.GetNumNz(), A_pages.GetNumPages(),
    A_pages.GetPtrToGpuRowPtrs(),A_pages.GetPtrToGpuColInd(), A_pages.GetPtrToGpuValues(), b_pages.GetPtrToGpuValues(), x_pages.GetPtrToGpuValues(),
    aiL_pages.GetNumNz(), aiL_pages.GetPtrToGpuRowPtrs(), aiL_pages.GetPtrToGpuColInd(), aiL_pages.GetPtrToGpuValues(), 
    aiU_pages.GetNumNz() ,aiU_pages.GetPtrToGpuRowPtrs(), aiU_pages.GetPtrToGpuColInd(), aiU_pages.GetPtrToGpuValues(),
    solver_results.GetPtrToGpuIterCount(), solver_results.GetPtrToGpuConvFlag() , solver_results.GetPtrToGpuIterResNorm());

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    OET = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - start)).count()/1000; 
    PST = (double)(std::chrono::duration_cast<std::chrono::microseconds>(mid - start)).count()/1000; 
    IST = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - mid)).count()/1000;    

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
void Batched_ILU_ISAI_Preconditioned_BiCGSTAB_Gpu(const std::vector<std::string> & subdir, const PagedCSRMatrices & A_pages,
    const PagedVectors& b_pages,PagedVectors & x_pages,const bool is_scaled,  SolverResults & solver_results , 
    const bool is_parilu , const int num_iter_par_ilu,  double & PST, double & IST, double & OET  )
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


    success_code = Batched_BiCGSTAB_ILU_ISAI_Gpu_helper(A_pages,b_pages,x_pages, solver_results, is_parilu, num_iter_par_ilu, PST, IST, OET);

    std::string solution_file;

    if(is_scaled == true)
        solution_file = "x_scaled_gpu_ilu_isai_bicgstab.mtx";
    else
        solution_file = "x_gpu_ilu_isai_bicgstab.mtx";

    if(success_code == 1)
    {
        x_pages.CopyFromGpuToCpu();
        Print_ans(subdir,x_pages, solution_file);
        std::cout << "files containing soluation: x  are produced...  ( " <<  solution_file <<  " ) in their respective directories " << std::endl;

    }


}
