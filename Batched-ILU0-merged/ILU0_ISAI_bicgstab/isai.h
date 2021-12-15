#pragma once


class PagedCSRMatrices;


enum class mat_type { general, lower_tri , upper_tri}; //not doing the spd case

void GenerateISAI_gpu(PagedCSRMatrices & aiA_pages,const PagedCSRMatrices & A_pages, const mat_type A_type, const int power);