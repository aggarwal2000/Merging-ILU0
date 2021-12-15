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


namespace {

#include "Bicgstab_common.hpp"

__device__ void legacy_sparse_lower_triangular_solve(const int num_rows, const int* const L_row_ptrs, const int* const L_col_idxs, 
const double* const L_values, const double* const vec_shared, volatile double* const temp_vec_shared)
{   
   
        const int row_index = threadIdx.x;

        if(row_index >= num_rows)
        {
            return;
        }

        double sum = 0;

        const int start = L_row_ptrs[row_index];
        const int end = L_row_ptrs[row_index + 1] - 1;
        int i = start;
        
        
        bool completed = false;

        while(!completed)
        {   
            

            const int col_index = L_col_idxs[i];

            if( i < end  &&  isfinite(temp_vec_shared[col_index]))
            {
                sum += L_values[i] * temp_vec_shared[col_index];
                i++;
            }

           
            if(i == end)
            {   
                temp_vec_shared[row_index] = (vec_shared[row_index] - sum)/L_values[end];
               
                completed = true;
               
            }

          
        }

       
        
}


__device__ void legacy_sparse_upper_triangular_solve(const int num_rows,  const int* const U_row_ptrs, const int* const U_col_idxs, 
const double* const U_values, volatile const double* const temp_vec_shared, volatile double* const vec_hat_shared)
{
    const int row_index = threadIdx.x;

    if(row_index >= num_rows)
    {
        return;
    }

    double sum = 0;

    const int start = U_row_ptrs[row_index];
    const int end = U_row_ptrs[row_index + 1]  - 1;
    int i = end;

    bool completed = false;

    while(!completed )
    {   
       

        const int col_index = U_col_idxs[i];

        if( i > start && isfinite(vec_hat_shared[col_index]))
        {
            sum += U_values[i] * vec_hat_shared[col_index];
            i--;
        }

      
        if(i == start)
        {
            vec_hat_shared[row_index] = (temp_vec_shared[row_index] - sum)/U_values[start];
           
            completed = true;
        }

      
    }

}

__device__ void ApplyPreconditionerILU(const int num_rows , const int* const L_row_ptrs, 
    const int* const L_col_idxs , const double* const L_values,  const int* const U_row_ptrs,
    const int* const U_col_idxs, const double* const U_values,  const double* const vec_shared,volatile double* const vec_hat_shared)
{

   
    // vec_hat = precond * vec
    // => L * U  * vec_hat = vec
    // => L * y = vec , find y , and then U * vec_hat = y, find vec_hat

    //sparse triangular solves

    //if we want to use the busy waiting while loop approach, then the num_rows should be <= threadblock size, else there is possibility of a deadlock !
    assert(num_rows <= blockDim.x);
    //TODO: For upper trsv, use thread 0 for the bottommost row, this way we could avoid :  assert(num_rows <= blockDim.x), as there won't be a possibility of deadlock then!

    __shared__  volatile double temp_vec_shared[MAX_NUM_ROWS];

    for(int i = threadIdx.x ; i < num_rows; i += blockDim.x)
    {
        temp_vec_shared[i] = 1.8/0; //TODO: find a better way to deal with this!
        vec_hat_shared[i] = 1.3/0;

    }

    __syncthreads();
    
    
    legacy_sparse_lower_triangular_solve(num_rows,  L_row_ptrs, L_col_idxs, L_values, vec_shared, temp_vec_shared);

    __syncthreads();

    legacy_sparse_upper_triangular_solve(num_rows,  U_row_ptrs, U_col_idxs, U_values, temp_vec_shared, vec_hat_shared);

}




__global__ void KernelBatchedPreconditionedBiCGSTAB_ILU(const int num_rows, const int num_nz, const int num_pages, const int* const row_ptrs, 
    const int* const col_inds, const double* const vals_mat, const double* const vals_rhs, double* const vals_ans,
    const int L_nnz , const int* const L_row_ptrs, const int* const L_col_idxs, const double* const L_vals ,
    const int U_nnz, const int* const U_row_ptrs, const int* const U_col_idxs, const double* const U_vals,
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
                        

                        ApplyPreconditionerILU(num_rows, L_row_ptrs, L_col_idxs, L_vals + page_id * L_nnz, U_row_ptrs, U_col_idxs , U_vals + page_id * U_nnz, p_shared, p_hat_shared);

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

                     
                        ApplyPreconditionerILU(num_rows, L_row_ptrs, L_col_idxs, L_vals + page_id * L_nnz, U_row_ptrs, U_col_idxs , U_vals + page_id * U_nnz, s_shared, s_hat_shared);
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




int Batched_BiCGSTAB_ILU_Gpu_helper(const PagedCSRMatrices & A_pages,const PagedVectors& b_pages,PagedVectors & x_pages, 
    SolverResults & solver_results,const bool is_parilu , const int app_num_conv_ilu , const int num_iter_par_ilu,  double & PST, double & IST, double & OET  )
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
        assert(1 <= app_num_conv_ilu);
        assert(app_num_conv_ilu <= 4);
        const int approach_num = app_num_conv_ilu;
        //Note: For pele matrices, approach 1 works better as compared to the depenedency graph approach as the matrices are not that sparse. For other cases, approach 3 is exepected to be faster than others.
        //std::cout << " \nilu " << std::endl;
        ILU_0_Factorization_Gpu(A_pages , L_pages, U_pages, approach_num);
    }
    
    cudaDeviceSynchronize();
    auto mid = std::chrono::high_resolution_clock::now();

    dim3 block(THREADS_PER_BLOCK,1,1);
    dim3 grid_solver(A_pages.GetNumPages(),1,1 );

    //------------------------------------------------------------------------------- Call main solver kernel-------------------------------------------------//

    KernelBatchedPreconditionedBiCGSTAB_ILU<<< grid_solver, block , 0  >>>(A_pages.GetNumRows(), A_pages.GetNumNz(), A_pages.GetNumPages(),
    A_pages.GetPtrToGpuRowPtrs(),A_pages.GetPtrToGpuColInd(), A_pages.GetPtrToGpuValues(), b_pages.GetPtrToGpuValues(), x_pages.GetPtrToGpuValues(),
    L_pages.GetNumNz(), L_pages.GetPtrToGpuRowPtrs(), L_pages.GetPtrToGpuColInd(), L_pages.GetPtrToGpuValues(), 
    U_pages.GetNumNz() ,U_pages.GetPtrToGpuRowPtrs(), U_pages.GetPtrToGpuColInd(), U_pages.GetPtrToGpuValues(),
    solver_results.GetPtrToGpuIterCount(), solver_results.GetPtrToGpuConvFlag() , solver_results.GetPtrToGpuIterResNorm());

    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    
    OET = (double)(std::chrono::duration_cast<std::chrono::microseconds>(stop - start)).count()/1000; 
    PST = (double)(std::chrono::duration_cast<std::chrono::microseconds>(mid - start)).count()/1000; 
    IST = (double)(std::chrono::duration_cast<std::chrono::microseconds>(stop- mid)).count()/1000;    

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
void Batched_ILU_Preconditioned_BiCGSTAB_Gpu(const std::vector<std::string> & subdir, const PagedCSRMatrices & A_pages,
    const PagedVectors& b_pages,PagedVectors & x_pages,const bool is_scaled,  SolverResults & solver_results ,
     const bool is_parilu , const int app_num_conv_ilu ,  const int num_iter_par_ilu,  double & PST, double & IST, double & OET   )
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


    success_code = Batched_BiCGSTAB_ILU_Gpu_helper(A_pages,b_pages,x_pages, solver_results, is_parilu, app_num_conv_ilu ,  num_iter_par_ilu, PST, IST, OET);

    std::string solution_file;
    std::string specifics;
    if(is_parilu == true)
	{
	   specifics = std::string("parILU_") + std::to_string(num_iter_par_ilu)  +  std::string("_");
	}
	else
	{
	   specifics = std::string("conv_ILU_app_") +  std::to_string(app_num_conv_ilu)  +  std::string("_");
	}

    if(is_scaled == true)
        solution_file = "x_scaled_gpu_" +  specifics +  "bicgstab.mtx";
    else
        solution_file = "x_gpu_" +  specifics +  "bicgstab.mtx";

    if(success_code == 1)
    {
        x_pages.CopyFromGpuToCpu();
        Print_ans(subdir,x_pages, solution_file);
        std::cout << "files containing soluation: x  are produced...  ( " <<  solution_file <<  " ) in their respective directories " << std::endl;

    }


}
