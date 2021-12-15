#pragma once
#include<vector>
#include<string>

class PagedCSRMatrices;
class PagedVectors;
class SolverResults;


#define MAX_ITER 500

#define ATOL 0.00000000001


void Batched_ILU_Preconditioned_BiCGSTAB_Gpu(const std::vector<std::string> & subdir, const PagedCSRMatrices & A_pages,
const PagedVectors& b_pages,PagedVectors & x_pages,const bool is_scaled,  SolverResults & solver_results ,
 const bool ,const int, const int, double &, double &, double & );

void Batched_ILU_ISAI_Preconditioned_BiCGSTAB_Gpu(const std::vector<std::string> & subdir, const PagedCSRMatrices & A_pages,const PagedVectors& b_pages,PagedVectors & x_pages,const bool is_scaled,  SolverResults & solver_results , const bool , const int, double &, double &, double & );

void Batched_General_ISAI_Preconditioned_BiCGSTAB_Gpu(const std::vector<std::string> & subdir, const PagedCSRMatrices & A_pages,
    const PagedVectors& b_pages,PagedVectors & x_pages,const bool is_scaled,  SolverResults & solver_results , 
     double & PST, double & IST, double & OET  );


void Batched_conv_ILU_app1_Preconditioned_BiCGSTAB_merged_Gpu(const std::vector<std::string> & subdir, const PagedCSRMatrices & A_pages,
    const PagedVectors& b_pages,PagedVectors & x_pages,const bool is_scaled,  SolverResults & solver_results  );
