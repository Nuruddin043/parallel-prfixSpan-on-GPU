
#include<string>
#include <sstream>
#include<ctime>
#include <chrono>
#include<fstream>
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include<stack>
#include<vector>
#include "frequent_items.h"
#include "projected_database.h"
#include "prefix_span.h"
using namespace std;
fstream _file;


vector < vector<int> > sequential_patterns;

int main() {
	int found, cnt = 0, total_row = 0;
	string line, temp;
	clock_t startt,endt;
	thrust::host_vector<int, thrust::cuda::experimental::pinned_allocator<int>> Hdata;
	thrust::host_vector<int, thrust::cuda::experimental::pinned_allocator<int>> Hstart;
	thrust::host_vector<int, thrust::cuda::experimental::pinned_allocator<int>> Hend;
	thrust::device_vector<int> device_data;
	thrust::device_vector<int> start(total_row);
	thrust::device_vector<int> end(total_row);
	freopen("data.out","w",stdout);

	ifstream file;
	file.open("MSNBC_SPMF.txt");
	if (!file) {
		cout << "file not found \n";
		return -1;
	}
//	auto startt = chrono::steady_clock::now();
	while (getline(file, line))
	{
		istringstream ss(line);
	//	int* ptr = raw_pointer_cast(&data[total_row]);
		Hstart.push_back(cnt);
		while (ss >> temp)
		{
			if (stringstream(temp) >> found) {
				Hdata.push_back(found);
				cnt++;
			}

		}
		total_row++;

		Hend.push_back(cnt);

	}
	file.close();


	device_data=Hdata;
	start=Hstart;
	end=Hend;
	int* dptr = raw_pointer_cast(&device_data[0]);
	int* startPtr = raw_pointer_cast(&start[0]);
	int* endPtr = raw_pointer_cast(&end[0]);
	startt = clock();
	prefix_Span(dptr, startPtr, endPtr, total_row,sequential_patterns);

//	auto endt = chrono::steady_clock::now();
//	auto diff = endt - startt;
	//cout << chrono::duration <double, milli>(diff).count() << " ms" << endl;
	endt = clock();
  double interval = (double)(endt - startt) / CLOCKS_PER_SEC;
  cout<<"cost time : " <<interval<<endl;
	for (int i = 0; i < sequential_patterns.size(); i++) {
		for (int j = 0; j < sequential_patterns[i].size(); j++) {
			cout << sequential_patterns[i][j] << " ";
		}
		cout << endl;
	}
	 	gpuErrchk(cudaDeviceReset());
}
