#include<iostream>
#include<stdio.h>
#include<vector>
#include<array>
#include<cassert>
#include<chrono>
#include<cmath>
#include "cuda_profiler_api.h"
#include "matrix.h"
#include "ReadWriteData.h"
#include "header.h"
#include "isai.h"
#include "cooperative_groups.h"

namespace{
    
#define row_size_limit 32
#define default_block_size 64

constexpr int subwarpgrp_size = 32;

__global__ void copy_array_kernel(const int size, const int* const src, int* const dst)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(gid < size)
    {
        dst[gid] = src[gid];
    }
}    

void copy_array(const int size, const int* const src_gpu, int* const dst_gpu)
{
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(ceil( (double)size / (double)THREADS_PER_BLOCK ));

    copy_array_kernel <<< grid, block >>> (size, src_gpu , dst_gpu);
}

void Allocate_Mem_and_Set_row_ptrs_and_col_idxs_approx_inverse_acc_to_chosen_sparsity_pattern( const PagedCSRMatrices & A_pages, 
    const int power, PagedCSRMatrices & aiA_pages )
{   
    assert(power >= 1);

    if(power == 1)
    {
        aiA_pages.SetNumRows(A_pages.GetNumRows());
        aiA_pages.SetNumCols(A_pages.GetNumCols());
        aiA_pages.SetNumNz(A_pages.GetNumNz());
        aiA_pages.SetNumPages(A_pages.GetNumPages());
        aiA_pages.AllocateMemory(LOCATION::GPU);
        copy_array(aiA_pages.GetNumRows() + 1, A_pages.GetPtrToGpuRowPtrs(), aiA_pages.GetPtrToGpuRowPtrs());
        copy_array( aiA_pages.GetNumNz() ,  A_pages.GetPtrToGpuColInd() , aiA_pages.GetPtrToGpuColInd());
    }
    else
    {
        //std::cout << "Have not implemented for power > 1, i.e. power =  " << power << std::endl;
        // exit(1);

        const int nrows = A_pages.GetNumRows();
        const int nnz_A = A_pages.GetNumNz();

        std::vector<int> row_ptrs_arr(nrows + 1);
        std::vector<int> col_idxs_arr(nnz_A);
        std::vector<double> values_arr(nnz_A);
        cudaMemcpy(row_ptrs_arr.data(), A_pages.GetPtrToGpuRowPtrs() , sizeof(int) * (nrows + 1), cudaMemcpyDeviceToHost);
        cudaMemcpy(col_idxs_arr.data(), A_pages.GetPtrToGpuColInd(), sizeof(int)* nnz_A, cudaMemcpyDeviceToHost);
        cudaMemcpy(values_arr.data(), A_pages.GetPtrToGpuValues(), sizeof(double)* nnz_A, cudaMemcpyDeviceToHost);
        const int* row_ptrs = row_ptrs_arr.data();
        const int* col_idxs = col_idxs_arr.data();
        const double* values = values_arr.data();

        std::vector<double> dense_second(nrows * nrows, 0); //store in col. major order (good hit rate - mult. kernel)


        for(int row = 0; row < nrows; row++)
        {   
            int st = row_ptrs[row];
            int end = row_ptrs[row + 1];

            for(int i = st; i < end; i++ )
            {
                int col = col_idxs[i];
                double val = values[i];
                dense_second[col * nrows + row] = val;
            }
        }


        std::vector<double> dense_ans(nrows * nrows, 0); // also in col. major order (good hit rate - mult. kernel-- as to be reused as dense_second)

        for(int p = 2; p <= power; p++)
        {       
                dense_ans = std::vector<double>(nrows * nrows, 0);

                for(int col = 0; col < nrows; col++)
                {

                    for(int row = 0; row < nrows; row++)
                    {
                        int st = row_ptrs[row];
                        int end = row_ptrs[row + 1];

                        double sum = 0;

                        for(int i = st; i < end; i++ )
                        {
                            int col_idx = col_idxs[i];
                            double val = values[i];
                            sum += val * dense_second[col * nrows + col_idx];
                        }

                        dense_ans[col * nrows + row] = sum;

                    }

                }


                dense_second = dense_ans;

        }

       

        int total_nz = 0;
        std::vector<int> temp_row_ptrs(nrows + 1, 0);
        //poor data locality/(cache hit rate)
        int nnz_per_row = 0;
        for(int row = 0; row < nrows; row++)
        {   
            nnz_per_row = 0;
            for(int col = 0; col < nrows; col++)
            {
                double val = dense_second[col * nrows + row];

                if(val != 0)
                {
                    total_nz++;
                    nnz_per_row++;
                }

            }

            temp_row_ptrs[row + 1] = temp_row_ptrs[row] + nnz_per_row;
        }


        aiA_pages.SetNumRows(A_pages.GetNumRows());
        aiA_pages.SetNumCols(A_pages.GetNumCols());
        aiA_pages.SetNumNz(total_nz);
        aiA_pages.SetNumPages(A_pages.GetNumPages());
        aiA_pages.AllocateMemory(LOCATION::CPU);

        int* aiA_row_ptrs = aiA_pages.GetPtrToCpuRowPtrs();
        int* aiA_col_idxs = aiA_pages.GetPtrToCpuColInd();
        
        for(int row = 0; row < nrows; row++)
        {   
            int ptr = temp_row_ptrs[row];
            aiA_row_ptrs[row] = ptr;

            for(int col = 0; col < nrows; col++)
            {
                double val = dense_second[col * nrows + row];

                 if(val != 0)
                {   
                    aiA_col_idxs[ptr] = col;
                    ptr++;
                }
            }
        }

        aiA_row_ptrs[nrows] = temp_row_ptrs[nrows];

        aiA_pages.CopyFromCpuToGpu();
    }
}

template< typename T>
__global__ void initialize_array_kernel(const T ival, T* const arr, const int size)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid < size)
    {
        arr[gid] = ival;
    }
}

template< typename T >
void initialize_array(const T ival, T* const arr_gpu, const int size)
{
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(ceil((double)size / (double)THREADS_PER_BLOCK));

    initialize_array_kernel<<< grid, block>>>(ival, arr_gpu , size);
}


void Initialize_pattern(int* const dense_mats_patterns_gpu , int* const rhs_one_idxs_gpu , const int nrows)
{
    initialize_array<int>(-1, dense_mats_patterns_gpu, nrows* row_size_limit * row_size_limit );
    initialize_array<int>(-1 , rhs_one_idxs_gpu, nrows);
}


template < int subwarpgrp_size >
__global__ void Extract_pattern_kernel_approach1(const int nrows, const int* const A_row_ptrs,const int* const A_col_idxs, const int nnz_A, 
    const int* const aiA_row_ptrs, const int* const aiA_col_idxs, const int nnz_aiA, int* const dense_mats_patterns, 
    int* const rhs_one_idxs, int* const sizes )
{  
    
    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    for(int i_row_idx = blockIdx.x ; i_row_idx < nrows; i_row_idx += gridDim.x)
    {
        __shared__ int nonzero_cols[row_size_limit];

        int i_st = aiA_row_ptrs[i_row_idx];
        int i_end =  aiA_row_ptrs[i_row_idx + 1];
        int i_size = i_end - i_st;
        assert(i_size <= row_size_limit);
        int* dense_ptr = dense_mats_patterns + row_size_limit*row_size_limit*i_row_idx;

        if(threadIdx.x == 0)
        {
            sizes[i_row_idx] = i_size;
        }

        for(int i = threadIdx.x + i_st ; i < i_end; i += blockDim.x) 
        {   
            //coalesced memory acessses
            int i_col_idx = aiA_col_idxs[i];
            nonzero_cols[i - i_st] = i_col_idx;

            if(i_col_idx == i_row_idx)
            {
                rhs_one_idxs[i_row_idx] = i - i_st; //aiA must be sorted...
            }    
        }

        __syncthreads();

        auto subwarpgrp = cooperative_groups::tiled_partition<subwarpgrp_size>(cooperative_groups::this_thread_block());

        int subgrpwarp_id_in_block = threadIdx.x / subwarpgrp_size;
        int total_num_subwarp_grps_in_block = blockDim.x / subwarpgrp_size;

        //TODO: Implement this in an efficient way by parallelizing the merge sort way of comparing 2 sorted arrays
        for(int q = subgrpwarp_id_in_block; q < i_size; q += total_num_subwarp_grps_in_block)
        {
            int m_row_idx = nonzero_cols[q];

            //the subwarp_grp deals with this row in M

            int id_in_subwarp_grp = subwarpgrp.thread_rank();

            int m_st = A_row_ptrs[m_row_idx];
            int m_end = A_row_ptrs[m_row_idx + 1];

            for(int i = id_in_subwarp_grp + m_st; i < m_end; i += subwarpgrp_size)
            {   
                
                int m_col_idx = A_col_idxs[i];

                for(int v = 0; v < i_size; v++)
                {   

                   if( nonzero_cols[v] == m_col_idx)
                   {    
                       //write into dense mat (row: q, col: v ); storage order of dense mat: row-major
                       //NOTE: Dense mat is not transposed here...
                       dense_ptr[q * row_size_limit + v] = i;    //CSR mat A must be sorted, also the non zero cols array

                       // if we want to store transpose, write at ( v,q), but then store matrix in column major order
                       break;
                   }
                }

            }

            subwarpgrp.sync(); //Remove
        }
        __syncthreads(); //Remove
    }
}





void Extract_Dense_Linear_Sys_pattern(const PagedCSRMatrices & aiA_pages,const PagedCSRMatrices & A_pages,
    int* const dense_mats_patterns_gpu, int* const rhs_one_idxs_gpu, int* const sizes_gpu)
{
    const int* const aiA_pages_row_ptrs_gpu = aiA_pages.GetPtrToGpuRowPtrs();
    const int* const aiA_pages_col_idxs_gpu = aiA_pages.GetPtrToGpuColInd();

    const int* const A_pages_row_ptrs_gpu = A_pages.GetPtrToGpuRowPtrs();
    const int* const A_pages_col_idxs_gpu = A_pages.GetPtrToGpuColInd();

    const int nrows = A_pages.GetNumRows();
    const int npages = A_pages.GetNumPages();

    const int nnz_A = A_pages.GetNumNz();
    const int nnz_aiA = aiA_pages.GetNumNz();

    Initialize_pattern(dense_mats_patterns_gpu, rhs_one_idxs_gpu, nrows);

    cudaProfilerStart();

    dim3 block1(THREADS_PER_BLOCK);

    dim3 grid1(nrows);

    Extract_pattern_kernel_approach1 < subwarpgrp_size > <<< grid1, block1 >>> (nrows, A_pages_row_ptrs_gpu, A_pages_col_idxs_gpu, nnz_A, aiA_pages_row_ptrs_gpu, 
        aiA_pages_col_idxs_gpu , nnz_aiA, dense_mats_patterns_gpu, rhs_one_idxs_gpu, sizes_gpu);
    
    cudaProfilerStop();
   
}


template <typename T >
__device__ double solve_upper_tri_dense(T subwarp_grp, double * const local_row , const int size)
{   
    const int local_id = subwarp_grp.thread_rank();

    double rhs = local_id == size -1 ? 1 : 0;
    
    double sol = rhs;

    for(int dense_col_idx = size - 1; dense_col_idx >= 0; dense_col_idx --)
    {
        const double ele = local_row[dense_col_idx];

        if(dense_col_idx == local_id)
        {   
            assert(ele != 0.0); //zero at diagonal position
            sol = sol/ele;
        }

        subwarp_grp.sync();

        const double bot = subwarp_grp.shfl(sol, dense_col_idx);

        if(local_id < dense_col_idx)
        {
            sol = sol - bot*ele;
        }

    }

    return sol;    
}


template <typename T >
__device__ double solve_lower_tri_dense(T subwarp_grp, double * const local_row , const int size)
{   
    const int local_id = subwarp_grp.thread_rank();

    double rhs = local_id == 0 ? 1 : 0;
    
    double sol = rhs;

    for(int dense_col_idx = 0; dense_col_idx < size; dense_col_idx++ )
    {
        const double ele = local_row[dense_col_idx];

        if(dense_col_idx == local_id)
        {   
            assert(ele != 0.0); //zero at diagonal position
            sol = sol/ele;
        }

        subwarp_grp.sync();

        const double top = subwarp_grp.shfl(sol, dense_col_idx);

        if(local_id > dense_col_idx)
        {
            sol = sol - top*ele;
        }

    }

    return sol;    
}


template < typename T >
__device__ int choose_pivot_row(T subwarpgrp, const int diag_pos,  double* const local_row, const int size)
{
    const double val_tid = local_row[diag_pos];

    int pivot_row_idx = diag_pos;

    double pivot = subwarpgrp.shfl(val_tid, diag_pos);

    for(int i = diag_pos + 1; i < size; i++)
    {   
        double temp = subwarpgrp.shfl(val_tid, i);
        if( fabs(pivot) < fabs(temp))
        {
            pivot = temp;
            pivot_row_idx = i;
        }
    }

    return pivot_row_idx;

}


template < typename T >
__device__ int choose_pivot_row_1(T subwarpgrp, const int diag_pos,  double* const local_row, const int size)
{   
   const int local_id =  subwarpgrp.thread_rank();
   int piv_row_idx = local_id;
   double val = local_row[diag_pos];
   double val1 = subwarpgrp.shfl(local_row[diag_pos], diag_pos);

   if(local_id >= size || local_id < diag_pos)
   {
       piv_row_idx = diag_pos;
       val = val1;
   }

   const int subwarp_size = subwarpgrp.size();

   for(int offset = subwarpgrp_size/2; offset > 0; offset = offset/2)
   {    
       subwarpgrp.sync();

       double val_other = subwarpgrp.shfl_down(val,offset );
       double piv_row_idx_other = subwarpgrp.shfl_down(piv_row_idx, offset);
       if( fabs(val_other) > fabs(val))
       {    
           val = val_other;
           piv_row_idx = piv_row_idx_other;
       }
   }

   //0th thread has correct piv_row_idx

   subwarpgrp.sync();
   piv_row_idx = subwarpgrp.shfl(piv_row_idx , 0);

   return piv_row_idx;
}



template < typename T >
__device__ void swap_rows_and_rhs(T subwarpgrp, const int diag_pos, const int pivot_row_idx, double* const local_row, const int size, double & rhs)
{
    const int local_id = subwarpgrp.thread_rank();

    for(int col = 0; col < size; col++)
    {
       double diag_tid_col_val = subwarpgrp.shfl(local_row[col], diag_pos);

       double piv_row_tid_col_val = subwarpgrp.shfl(local_row[col], pivot_row_idx);

       if(local_id == diag_pos)
       {
           local_row[col] = piv_row_tid_col_val;
       }

       if(local_id == pivot_row_idx)
       {
           local_row[col] = diag_tid_col_val;
       }
    }

    double diag_tid_rhs = subwarpgrp.shfl(rhs, diag_pos);

    double piv_row_tid_rhs = subwarpgrp.shfl(rhs, pivot_row_idx);

    if(local_id == diag_pos)
    {
        rhs = piv_row_tid_rhs;
    }

    if(local_id == pivot_row_idx)
    {
        rhs = diag_tid_rhs;
    }

}


template < typename T >
__device__ void RowTransformation( T subwarpgrp ,const int diag_pos, double* const local_row,const int size,
     double & rhs)
{
    const int local_id = subwarpgrp.thread_rank();
    const double diag_ele = subwarpgrp.shfl(local_row[diag_pos], diag_pos);
    if(diag_ele == 0)
    {
        printf("\n Got 0 at diag position while solving general dense linear system...\n");
        //assert(0);
    }
    const double multiplier = local_row[diag_pos]/diag_ele;

    const double rhs_key_val = subwarpgrp.shfl(rhs, diag_pos);

    for(int col = 0; col < size; col++)
    {    
        const double col_key_val =  subwarpgrp.shfl(local_row[col], diag_pos);

        if(local_id != diag_pos)
        {
            local_row[col] -= multiplier * col_key_val;
        }
      
    }

    if(local_id != diag_pos)
    {
        rhs -= multiplier * rhs_key_val;
    }

}


template < typename T >
__device__ double solve_general_dense(T subwarpgrp, double* const local_row,const int size, const int rhs_one_idx)
{   
    const int local_id = subwarpgrp.thread_rank();

    double rhs = rhs_one_idx == local_id ? 1 : 0;

    for(int diag_pos = 0; diag_pos < size; diag_pos++)
    {
        const int pivot_row_idx = choose_pivot_row_1(subwarpgrp, diag_pos, local_row, size);

        if(pivot_row_idx != diag_pos)
        {
            swap_rows_and_rhs(subwarpgrp, diag_pos, pivot_row_idx, local_row, size, rhs);
        }
      
        subwarpgrp.sync();

        RowTransformation( subwarpgrp , diag_pos, local_row, size, rhs);

        subwarpgrp.sync();
    }

    rhs = rhs/local_row[local_id];

    return rhs;
}



template < int subwarpgrp_size >
__global__ void  fill_values_and_solve_kernel_approach1( const int npages, const int nrows, int* const aiA_row_ptrs, 
    int* const aiA_col_idxs, double* const aiA_values , const int nnz_aiA, const int* const A_row_ptrs, const int* const A_col_idxs, 
    const double* const A_values, const int nnz_A , const int* const dense_patterns, const int* const rhs_one_idxs, const int* const sizes, const int matrix_type )
{   
    static_assert(row_size_limit <= subwarpgrp_size, "incompatible warp size");

    auto subwarpgrp = cooperative_groups::tiled_partition<subwarpgrp_size>(cooperative_groups::this_thread_block());

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int subwarp_id_in_grid = gid/subwarpgrp_size;
    const int total_num_subwarp_grps_in_grid = (gridDim.x * blockDim.x)/subwarpgrp_size;
    const int local_id = subwarpgrp.thread_rank();
    
    for(int i = subwarp_id_in_grid; i < nrows * npages ; i += total_num_subwarp_grps_in_grid)
    {
        
        int row = i % nrows;
        int page_id = i/nrows;

        const int* dense_ptr = dense_patterns + row_size_limit*row_size_limit*row;
        int size = sizes[row];
        int rhs_one_idx = rhs_one_idxs[row];

        double local_row[row_size_limit];
        
        //implicit transpose
        for(int k = 0; k < size; k++) 
        {   
            int val_idx = dense_ptr[k * row_size_limit + local_id]; //coalesced accesses by subwarp_grp

            if(val_idx != -1)
            {
                local_row[k] = A_values[val_idx + page_id * nnz_A];
            }
            else
            {
                local_row[k] = 0;
            } 
        }

        subwarpgrp.sync();

        double ai_A_ele;
        //Now solve the system
        if( matrix_type == 0 ) //i.e. lower
        {
            ai_A_ele = solve_upper_tri_dense(subwarpgrp, local_row, size);
        }
        else if(matrix_type == 1) //i.e. upper
        {
            ai_A_ele = solve_lower_tri_dense(subwarpgrp , local_row, size);
        }
        else if(matrix_type == 2) //i.e. dense
        {   
            ai_A_ele = solve_general_dense(subwarpgrp, local_row, size, rhs_one_idx);
        }
        else{
            printf("\n No such case: line: %d and file: %s " , __LINE__ , __FILE__);
            assert(false);
        }

        if(local_id < size)
        {
            aiA_values[page_id * nnz_aiA + aiA_row_ptrs[row] + local_id] = ai_A_ele; //coalesced access by subwarp_grp while writing
        }
     
    }
}



void Fill_values_dense_mat_and_solve(PagedCSRMatrices & aiA_pages, const PagedCSRMatrices & A_pages , 
    const int* const dense_mats_patterns_gpu, const int* const rhs_one_idxs_gpu, const int* const sizes_gpu, enum mat_type A_type)
{
    int* const aiA_pages_row_ptrs_gpu = aiA_pages.GetPtrToGpuRowPtrs();
    int* const aiA_pages_col_idxs_gpu = aiA_pages.GetPtrToGpuColInd();
    double* const aiA_pages_values_gpu = aiA_pages.GetPtrToGpuValues();

    const int* const A_pages_row_ptrs_gpu = A_pages.GetPtrToGpuRowPtrs();
    const int* const A_pages_col_idxs_gpu = A_pages.GetPtrToGpuColInd();
    double* const A_pages_values_gpu = A_pages.GetPtrToGpuValues();

    const int nrows = A_pages.GetNumRows();
    const int npages = A_pages.GetNumPages();

    const int nnz_A = A_pages.GetNumNz();
    const int nnz_aiA = aiA_pages.GetNumNz();

    int matrix_type;

    if(A_type == mat_type::lower_tri)
    {
        matrix_type = 0;
    }
    else if(A_type == mat_type::upper_tri)
    {
        matrix_type = 1;
    }
    else if(A_type == mat_type::general)
    {
        matrix_type = 2;
    }
    else
    {
        std::cout << " No such case " << __LINE__  << "  and " << __FILE__ << std::endl;
    }

    cudaProfilerStart();

    dim3 block1(THREADS_PER_BLOCK);
    
    dim3 grid1(ceil( (double)subwarpgrp_size * nrows * npages/ (double)THREADS_PER_BLOCK ));

    //use one subwarp per dense linear system solve
    fill_values_and_solve_kernel_approach1 < subwarpgrp_size > <<< grid1, block1 >>>(npages, nrows, aiA_pages_row_ptrs_gpu, aiA_pages_col_idxs_gpu,
        aiA_pages_values_gpu , nnz_aiA, 
    A_pages_row_ptrs_gpu, A_pages_col_idxs_gpu,  A_pages_values_gpu , nnz_A, dense_mats_patterns_gpu, rhs_one_idxs_gpu, sizes_gpu, matrix_type );
    

    cudaProfilerStop();
}


void Print_ISAI(const PagedCSRMatrices & aiA_pages)
{   
    std::cout << "\n Printing aiA : " << std::endl;
    const int npages = aiA_pages.GetNumPages();
    const int nrows = aiA_pages.GetNumRows();
    const int nnz = aiA_pages.GetNumNz();
    const int* const aiA_row_ptrs = aiA_pages.GetPtrToCpuRowPtrs();
    const int* const aiA_col_idxs = aiA_pages.GetPtrToCpuColInd();
    const double* const aiA_values = aiA_pages.GetPtrToCpuValues();

    std::cout << "\n Row pointers: " << std::endl;
    for(int i = 0; i < nrows + 1; i++)
    {
        std::cout << aiA_row_ptrs[i] << "  ";
    }

    std::cout << "\n Col idxs: " << std::endl;
    for(int i = 0; i < nnz; i++)
    {
        std::cout << aiA_col_idxs[i] << "  ";
    }

    for(int page_id = 0; page_id < aiA_pages.GetNumPages(); page_id++)
    {
        std::cout << "\n Values for page: " << page_id << " are: " << std::endl;

        for(int i = 0; i < nnz; i++)
        {
            std::cout << aiA_values[ i + page_id * nnz] << "  ";
        }

    }
}

}//unnamed namespace



void GenerateISAI_gpu(PagedCSRMatrices & aiA_pages,const PagedCSRMatrices & A_pages, const mat_type A_type, const int power)
{   
   // printf("\n\n\n GPU ISAI \n\n\n");

    assert(A_pages.ExistsGPU() == true);

    const int npages = A_pages.GetNumPages();
    const int nrows = A_pages.GetNumRows();

    Allocate_Mem_and_Set_row_ptrs_and_col_idxs_approx_inverse_acc_to_chosen_sparsity_pattern(A_pages, power, aiA_pages);

    int* dense_mats_patterns_gpu = nullptr;
    int* rhs_one_idxs_gpu = nullptr;
    int* sizes_gpu = nullptr;

    cudaMalloc( (void**)&dense_mats_patterns_gpu, sizeof(int) * row_size_limit * row_size_limit * nrows);
    cudaMalloc( (void**)&rhs_one_idxs_gpu , sizeof(int) * nrows );
    cudaMalloc( (void**)& sizes_gpu, sizeof(int) * nrows);

    Extract_Dense_Linear_Sys_pattern(aiA_pages, A_pages, dense_mats_patterns_gpu , rhs_one_idxs_gpu , sizes_gpu);
    //For other matrices, sparsity pattern of |A0|^k is used, which may not be exact sp(|Ai|^k).
    
    Fill_values_dense_mat_and_solve(aiA_pages, A_pages, dense_mats_patterns_gpu, rhs_one_idxs_gpu, sizes_gpu, A_type);

    //aiA_pages.CopyFromGpuToCpu();

    //Print_ISAI(aiA_pages);
    
    cudaFree(dense_mats_patterns_gpu);
    cudaFree(rhs_one_idxs_gpu);
    cudaFree(sizes_gpu);

}

