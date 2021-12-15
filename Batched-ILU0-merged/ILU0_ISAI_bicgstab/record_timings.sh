#!/bin/bash

nvcc factorization.cu ILU_0_gpu.cu parILU_0_gpu.cu isai_gpu.cu  matrix.cu mmio.cpp  ReadWriteData.cpp SolverResults.cu PrecondBiCGSTAB_ilu_gpu.cu main_bicgstab_ilu.cpp -o ilu   

for category in  LF10 pores_1 mesh1em1 bcsstk01
do
	
	file_name="timings_conv_ilu0_app1_sep_"$category"_.txt"

	for ((problem_size= 100 ; problem_size<=10000 ; problem_size+=100))
	do

		./ilu $category $problem_size 0  0  1 0 $file_name
		
	done

	
done


nvcc factorization.cu ILU_0_gpu.cu parILU_0_gpu.cu isai_gpu.cu  matrix.cu mmio.cpp  ReadWriteData.cpp SolverResults.cu PrecondBiCGSTAB_exact_ilu0_app1_merged_gpu.cu main_bicgstab_ilu_merged.cpp -o ilu_merged   

for category in  LF10 pores_1 mesh1em1 bcsstk01
do
	
	file_name="timings_conv_ilu0_app1_merged_"$category"_.txt"

	for ((problem_size= 100 ; problem_size<=10000 ; problem_size+=100))
	do

		./ilu_merged $category $problem_size 0  $file_name
		
	done

	
done


