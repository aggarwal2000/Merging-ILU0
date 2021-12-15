#pragma once
#include<vector>
#include<string>

class PagedCSRMatrices;

void ParILU_0_Factorization_Gpu(const PagedCSRMatrices & A_pages , PagedCSRMatrices & L_pages, PagedCSRMatrices & U_pages, const int num_iterations);
