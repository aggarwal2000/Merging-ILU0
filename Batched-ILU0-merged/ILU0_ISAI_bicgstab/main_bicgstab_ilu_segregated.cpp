#include<iostream>
#include<vector>
#include<string>
#include <fstream>

#include"matrix.h"
#include"ReadWriteData.h"
#include"PrecondBiCGSTAB.h"
#include "SolverResults.h"



int main(int argc, char *argv[])
{	
	//only for conv ilu0
	if(argc < 4)
	{
		printf("\nFormat: ./a.out [category name] [problemsize] [1 for scaled]  [<optional> file to store execution time(cuda implementation)]\n");
		exit(0);
	}
	
	std::string category = argv[1];



	std::string dir = "../../../Pele_Matrices_market/" + category + "/";

	std::vector<std::string> subdir{};

	FillSubdir(dir,subdir);

	std::cout << "\nNumber of small problems in category - " << category << " is: " << subdir.size() << std::endl;

	int Problem_Size = std::stoi(argv[2]);

	std::cout << "\nSo, the number of small problems to be solved: " << Problem_Size*subdir.size() << std::endl;


	bool is_scaled = true;
	
	int scale_info = std::stoi(argv[3]);
	
	if(scale_info == 1)
	  is_scaled = true;
	else
	  is_scaled = false;


	// for(int i=0;i< subdir.size();i++)
	// 	std::cout << subdir[i] << std::endl;

	

	std::cout << "\n\nStart reading the matrices and their rhs..." << std::endl;

	PagedCSRMatrices A_pages;
	PagedVectors b_pages;

	ReadData(Problem_Size ,subdir,A_pages,b_pages, is_scaled); //store all info in cpu arrays.

	PagedVectors x_pages;
	x_pages.SetNumPages(A_pages.GetNumPages());
	x_pages.SetNumElements(A_pages.GetNumCols());
	x_pages.AllocateMemory(LOCATION::CPU);


	SolverResults solver_results(A_pages.GetNumPages(), ATOL, CPU_EXISTENCE::EXISTENT, GPU_EXISTENCE::EXISTENT);

	//solving A*x = b; 
	
	
	//---------------------------------------------------------------------------------------------------------------------------------------------------------


	std::cout << "***************************************************************************************************************************" << std::endl;
	std::cout << "***************************************************************************************************************************" << std::endl;
	std::cout << std::endl << "Welcome to ilu preconditioned BiCGSTAB \n";
	//Copy stuff to gpu
	A_pages.AllocateMemory(LOCATION::GPU);
	A_pages.CopyFromCpuToGpu();
	b_pages.AllocateMemory(LOCATION::GPU);
	b_pages.CopyFromCpuToGpu();
	x_pages.AllocateMemory(LOCATION::GPU);

	const int rounds = 5;

	double av_time = 0;

	
	for(int i=0; i< rounds; i++)
	{
		Batched_conv_ILU_app1_Preconditioned_BiCGSTAB_segregated_Gpu(subdir, A_pages, b_pages, x_pages, is_scaled,  solver_results);
		 
		av_time += solver_results.GetTimeTaken();
	
	}
	
	av_time = av_time/rounds;

	solver_results.CopyFromGpuToCpu();

	if(argc == 5)
	{
		std::string f = argv[4];
		std::ofstream timings_file;
  		timings_file.open (f, std::ofstream::app);
  		// timings_file << "category: "  <<  category << "   is_scaled: " << is_scaled << "   problem_size: " << Problem_Size << "  time(cuda) in millisec: " << solver_results.GetTimeTaken() <<  " \n";
		timings_file  << Problem_Size  << "  " << av_time <<   " \n";   
		timings_file.close();
	}
	
	std::string specifics;
	specifics = std::string("conv_ILU_app1_segregated_");
	
	PrintSolverResults(subdir,solver_results, std::string("Results_ilu_bicgstab_")  + specifics + category + std::string(".txt"));




}
