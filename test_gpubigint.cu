/*
文件名:test_gpubigint.cu
功能简介：
本文件用于测试自定义的 GPUBigInt 大整数类在 CUDA GPU 设备端的高精度运算能力。
主要测试内容包括：大整数的加法、减法、乘法、除法、取模、乘法逆元等操作，
并验证这些操作在 GPU 上的正确性和可用性。
GPUBigInt类适用于 CUDA 环境下的大整数算法开发与验证。
最后修改: 2025/06/05
*/
#include <iostream>
#include <time.h>
#include <iostream>
#include <chrono>
#include <random>
#include <gmp.h>
#include "cuda_bigint.h"

using namespace std;
__global__ void gpu_bigint_test(GPUBigInt* ret){
    //测试加法
    GPUBigInt curve_p = GPUBigInt("fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f",16); 
    GPUBigInt curve_n = GPUBigInt("115792089237316195423570985008687907852837564279074904382605163141518161494337",10);
    GPUBigInt gx = GPUBigInt("55066263022277343669578718895168534326250603453777594175500187360389116729240",10);
    GPUBigInt gy = GPUBigInt("483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8",16);


    GPUBigInt bint_a = GPUBigInt("23421345123534534452134123412341234234234213421342134213423414"); // 假设大整数使用8个32位整数表示
	GPUBigInt bint_b = GPUBigInt("2"); // 假设大整数使用8个32位整数表示

	
    GPUBigInt bint_c = ((bint_a * bint_b) + (curve_p/bint_b)) % curve_n;
    char tmp_str1[bit_length] = {0},tmp_str2[bit_length] = {0},tmp_str3[bit_length] = {0};
    bint_c.to_char(tmp_str1);
    printf("GPUBigInt四则运算的结果:\n\t%s\n",tmp_str1);
    
    bint_a = bint_c;
    bint_a.invert(curve_n.value, bint_c.value);
    bint_a.to_char(tmp_str1);
    curve_n.to_char(tmp_str2,16);
    bint_c.to_char(tmp_str3,10);
    printf("求乘法逆元:\n\t%s \n\t(mod %s)\n\t的乘法逆元inv = %s\n",tmp_str1,tmp_str2,tmp_str3);
    *ret = bint_c;
}
int main()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	for(int device=0; device<deviceCount; ++device)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp,device);
		cout<<"本机安装的GPU:"<<deviceProp.name<<endl;
	}
    if(deviceCount < 1)
    {
        cout << "本机没有检测到支持CUDA的GPU设备,请检查环境配置后重试。" << endl;
        return 1;
    }
    GPUBigInt* d_result;
    GPUBigInt h_result;
    cudaMalloc(&d_result, sizeof(GPUBigInt));
    gpu_bigint_test<<<1,1>>>(d_result);
    cudaMemcpy(&h_result, d_result, sizeof(GPUBigInt), cudaMemcpyDeviceToHost);
    cout << "GPU返回结果:" << h_result.to_string() << endl;
    cudaFree(d_result);
    return 0;    
}