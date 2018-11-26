#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkAffineTransform.h"
#include "itkMetaImageIO.h"
#include "itkHessianRecursiveGaussianImageFilter.h"
#include "itkSymmetricEigenAnalysis.h"
#include "itkSymmetricSecondRankTensor.h"
#include "itkCastImageFilter.h"
#include "itkVariableSizeMatrix.h"
#include <vector>
#include <iostream>
#include "itkRegionOfInterestImageFilter.h"
#include "itkExtractImageFilter.h"


int main(int argc, char * argv[])
{
	if (argc < 2)
	{
		std::cerr << "Usage: " << std::endl;
		std::cerr << argv[0] << "  inputImageFile" << std::endl;
		return EXIT_FAILURE;
	}
	
	//ROI
	const	unsigned int   Dimension = 3;
	typedef   double  PixelType;
	typedef itk::Image< PixelType, Dimension >   ImageType;
	typedef itk::ImageFileReader< ImageType  >  ReaderType;
	typedef itk::ImageFileWriter< ImageType >  WriterType;


	// get size
	typedef float       InputPixelType;
	typedef itk::Matrix< InputPixelType, Dimension, Dimension > MatrixType;
	typedef itk::Image< MatrixType, Dimension > MatrixImageType;
	typedef MatrixImageType::Pointer        MatrixImagePointer;
	typedef MatrixImageType::RegionType     RegionType;

	ReaderType::Pointer reader = ReaderType::New();
	WriterType::Pointer writer = WriterType::New();
	reader->SetImageIO(itk::MetaImageIO::New());
	reader->SetFileName(argv[1]);
	reader->Update();

	// Allocate the output image
	ReaderType::OutputImageType::Pointer input = reader->GetOutput();
	RegionType region = input->GetLargestPossibleRegion();
	MatrixImagePointer m_Output = MatrixImageType::New();
	m_Output->SetRegions(region);
	

	
	typedef itk::ExtractImageFilter<ImageType, ImageType> ROIImageType;
	ROIImageType::Pointer roi = ROIImageType::New();

	ImageType::IndexType start;
	start[0] = 0;
	start[1] = 0;
	start[2] = 0;
//	start[0] = 55;
//	start[1] = 55;
//	start[2] = 95;

	//int patch_size = 17;
	std::cout << region << std::endl;

	ImageType::SizeType size;
	size[0] = region.GetSize(0);
	size[1] = region.GetSize(1);
	size[2] = region.GetSize(2);
	//size[0] = size[1] = size[2] = patch_size;


	ImageType::RegionType desiredRegion(start, size);

	roi->SetInput(reader->GetOutput());
	roi->SetExtractionRegion(desiredRegion);

	std::string output_path = (argv[2]);
	writer->SetInput(roi->GetOutput());
	writer->SetImageIO(itk::MetaImageIO::New());
	writer->SetFileName(output_path);
	writer->Update();


	//hessian filter
	int i, j,k=0,x,y,z;
	double sigma,sline;
	const double sigma_start = 0.5;
	const double sigma_end = 5.0;
	const double sigma_step = 0.1;
	const int N = (int)((sigma_end - sigma_start + 1.0) / sigma_step);
	itk::VariableSizeMatrix<PixelType>  data_matrix;	data_matrix.SetSize(size[0]*size[1]*size[2], Dimension + 15);
	itk::VariableSizeMatrix<PixelType>  result_matrix;	result_matrix.SetSize(10000, Dimension + 11);
	typedef   itk::SymmetricSecondRankTensor<PixelType>	TensorPixelType;
	typedef   itk::HessianRecursiveGaussianImageFilter<ImageType >	HessianFilterType;
	typedef	  itk::Image<TensorPixelType, Dimension> TensorImageType;
	typedef   itk::Vector< PixelType, Dimension >  EigenValueVectorType;
	typedef   itk::Matrix< PixelType, Dimension, Dimension>	EigenVectorMatrixType;
	typedef   itk::SymmetricEigenAnalysis<TensorPixelType, EigenValueVectorType, EigenVectorMatrixType>	EigenAnalysisType;

	for (sigma = sigma_start, j = 0; sigma <= sigma_end; sigma += sigma_step, j++)
	{

		HessianFilterType::Pointer hessianFilter = HessianFilterType::New();
		hessianFilter->SetInput(roi->GetOutput());
		hessianFilter->SetSigma(sigma);
		hessianFilter->Update();
		TensorImageType::Pointer hessianimage = hessianFilter->GetOutput();

		EigenAnalysisType	eigenanalysis;
		eigenanalysis.SetDimension(Dimension);
		eigenanalysis.SetOrderEigenMagnitudes(true);

		typedef itk::ImageRegionIterator<TensorImageType> TensorIteratorType;
		TensorIteratorType eigenImageIt(hessianimage, hessianimage->GetLargestPossibleRegion());

		EigenValueVectorType eigenvalue;
		EigenVectorMatrixType eigenvector;

		for (eigenImageIt.GoToBegin(), i = 0; !eigenImageIt.IsAtEnd(); ++eigenImageIt, i++)
		{
			//Each row of the matrix 'EigenVectors' represents an eigen vector.
			eigenanalysis.ComputeEigenValuesAndVectors(eigenImageIt.Get(), eigenvalue, eigenvector);

			//sline
			sline = pow(sigma, 2.0) * (eigenvalue[0] - eigenvalue[1]);
			if (sline < 0.0){ sline = 0.0; }

			if (data_matrix(i, 13) <= sline)
			{
				z = (int)(i / (size[0] * size[1]));
				y = (int)((i - size[0] * size[1] * z) / size[0]);
				x = i - z*size[0] * size[1] - y*size[0];

				data_matrix(i, 0) = i;
				data_matrix(i, 1) = eigenvector(2, 0); //x1
				data_matrix(i, 2) = eigenvector(2, 1); //y1
				data_matrix(i, 3) = eigenvector(2, 2); //z1
				data_matrix(i, 4) = eigenvector(1, 0); //x2
				data_matrix(i, 5) = eigenvector(1, 1); //y2
				data_matrix(i, 6) = eigenvector(1, 2); //z2
				data_matrix(i, 7) = eigenvector(0, 0); //x3
				data_matrix(i, 8) = eigenvector(0, 1); //y3
				data_matrix(i, 9) = eigenvector(0, 2); //z3
				data_matrix(i, 10) = eigenvalue[2]; //λ1
				data_matrix(i, 11) = eigenvalue[1]; //λ2
				data_matrix(i, 12) = eigenvalue[0]; //λ3
				data_matrix(i, 13) = sline; //線状度
				data_matrix(i, 14) = sigma; //σ
				data_matrix(i, 15) = x; //x
				data_matrix(i, 16) = y; //y
				data_matrix(i, 17) = z; //z
				
			}
		}
		
	}
	
	double max = INT_MIN;  /* 最大値の変数を、可能性のある最小の値で初期化 */
	for (j = 0; j < size[0] * size[1] * size[2]; ++j) {
		if (max < data_matrix(j, 13)) {
			max = data_matrix(j, 13);
		}
	}

	for (int i = 0; i < size[0]*size[1]*size[2]; i++){
		if ((0 < data_matrix(i, 15)) && (data_matrix(i, 15) < size[0]) && (0 < data_matrix(i, 16)) && (data_matrix(i, 16) < size[1])
			&& (0 < data_matrix(i, 17)) && (data_matrix(i,17) < size[2]) && (data_matrix(i, 13) >= max)){
			result_matrix(k, 0) = data_matrix(i, 13); //sline
			result_matrix(k, 1) = data_matrix(i, 14); //sigma
			result_matrix(k, 2) = data_matrix(i, 1); //x1
			result_matrix(k, 3) = data_matrix(i, 2); //y1
			result_matrix(k, 4) = data_matrix(i, 3); //z1
			result_matrix(k, 5) = data_matrix(i, 4); //x2
			result_matrix(k, 6) = data_matrix(i, 5); //y2
			result_matrix(k, 7) = data_matrix(i, 6); //z2
			result_matrix(k, 8) = data_matrix(i, 7); //x3
			result_matrix(k, 9) = data_matrix(i, 8); //y3
			result_matrix(k, 10) = data_matrix(i, 9); //z3
			result_matrix(k, 11) = data_matrix(i, 15); //x
			result_matrix(k, 12) = data_matrix(i, 16); //y
			result_matrix(k, 13) = data_matrix(i, 17); //z
			k++;
		}
	}
	
	//output_txt
	std::ofstream textout(output_path+"result.txt", std::ios::out | std::ios::trunc);
	if (textout){
		textout << data_matrix;
		textout.close();
	}else{
		std::cerr << "Error" << std::endl;
	}
	
	std::ofstream textout2(output_path+"result2.txt", std::ios::out | std::ios::trunc);
	if (textout2){
		textout2 << result_matrix;
		textout2.close();
	}else{
		std::cerr << "Error" << std::endl;
	}

	
	return EXIT_SUCCESS;

}
