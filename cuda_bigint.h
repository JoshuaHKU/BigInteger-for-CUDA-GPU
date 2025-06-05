/*
File Name cuda_bigint.h
Function Description:A lightweight CUDA library for high-performance big integer arithmetic on GPUs.​​ 
Provides core operations (addition, subtraction, multiplication, division) and essential cryptographic 
primitives including modular exponentiation, multiplicative inverse (modular), and bitwise shifts.
 Last Modified: 2025-06-05
文件名: cuda_bigint.h
功能说明：可在CUDA架构GPU设备端独立运行的大整数运算函数集合，实现了大整数的加、减、乘、除等基本运算，
以及幂模、求乘法逆元、位移等加密领域常用的运算。
最后修改：2025-06-05
*/

#pragma once
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdlib>
#include <ctime>


const int BLOCK_BIT_SIZE=32;
const long long BLOCK_MASK=0xFFFFFFFFLL;
const long long BLOCK_MAX=0x100000000LL;
const int bit_length=256;					// 大整数的最大位数
const int block_length = 16;				// 每个大整数的最大块数


__device__ __host__ int get_real_length(const unsigned int a[],const int real_length)
{
	for(int i=real_length-1;i>=0;i--)
		if(a[i]!=0)
			return i+1;
	return 0;
}

__device__ __host__ int get_real_length(const unsigned long long a[],const int real_length)
{
	for(int i=real_length-1;i>=0;i--)
		if(a[i]!=0)
			return i+1;
	return 0;
}

__device__ __host__ int get_real_length(const int a[],const int real_length)
{
	for(int i=real_length-1;i>=0;i--)
		if(a[i]!=0)
			return i+1;
	return 0;
}
class ShareMemoryManage
{
public:
	void* operator new(size_t len)
	{
		void* ptr;
		cudaMallocManaged(&ptr,len);
		cudaDeviceSynchronize();
		return ptr;
	}

	void* operator new[](size_t len)
	{
		void* ptr;
		cudaMallocManaged(&ptr,len);
		cudaDeviceSynchronize();
		return ptr;
	}

	void operator delete(void* ptr)
	{
		cudaDeviceSynchronize();
		cudaFree(ptr);
	}

	void operator delete[](void* ptr)
	{
		cudaDeviceSynchronize();
		cudaFree(ptr);
		cudaDeviceSynchronize();
	}
};
/* 结构体 `CUDA_GPUBigInt` 是用于表示和操作大整数的核心数据结构，支持在CPU和GPU（CUDA设备端）上使用。
它通过一个整型数组 `blocks` 存储大整数的各个数位，支持最高4096位（可配置），并用 `real_length` 记录
实际有效位数，`sign` 表示正负号。该结构体内置了多种初始化、赋值、打印、字符串转换、随机数生成等方法，
能够方便地进行大整数的基本运算和数据管理，适用于高性能并行计算场景下的大整数运算需求。 */
struct CUDA_GPUBigInt:public ShareMemoryManage
{
	bool sign = true;				//负数为false，正数为true
	int real_length;
	int blocks[block_length];

	__device__ __host__ constexpr char get_dights(int i)
	{
		char DIGHTS[38]="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
		return DIGHTS[i];
	}

	__host__ void to_string(char* str,const int blocks[],const bool sign,const int radix,const int real_length)
	{
		int len=0;
		if(!sign)
			str[len++]='-';
		unsigned int* rest=new unsigned int[real_length];
		unsigned int* target=new unsigned int[real_length];
		for(int i=0;i<real_length;++i)
			target[i]=(unsigned int)blocks[i];
		while(true)
		{
			unsigned long long temp=0;
			unsigned long long rest_dight_sum=0;
			for(int i=real_length-1;i>=0;i--)
			{
				temp=(temp<<BLOCK_BIT_SIZE)|target[i];
				rest[i]=(unsigned int)(temp/radix);
				temp=temp%radix;
				rest_dight_sum=rest_dight_sum+rest[i];
			}
			str[len++]=get_dights((int)temp);
			if(rest_dight_sum==0)
				break;
			memcpy(target,rest,sizeof(unsigned int)*real_length);
		}
		int start_pos=0;
		if(!sign)
			start_pos=1;
		for(int i=start_pos;i<(len+1)/2;++i)
		{
			char x=str[len-i-1+start_pos];
			str[len-i-1+start_pos]=str[i];
			str[i]=x;
		}
		str[len++]='\0';
		delete[] target;
		delete[] rest;
	}
	//把大整数转换为10进制字符串
	__device__ __host__ void to_str(char* str, const int blocks[],const int real_length,const bool sign = true,const int radix=10)
	{
		int len=0;
		if(!sign)
			str[len++]='-';
		unsigned int rest[bit_length];
		unsigned int target[bit_length];
		for(int i=0;i<real_length;++i)
			target[i]=(unsigned int)blocks[i];
		while(true)
		{
			unsigned long long temp=0;
			unsigned long long rest_dight_sum=0;
			for(int i=real_length-1;i>=0;i--)
			{
				temp=(temp<<BLOCK_BIT_SIZE)|target[i];
				rest[i]=(unsigned int)(temp/radix);
				temp=temp%radix;
				rest_dight_sum=rest_dight_sum+rest[i];
			}
			str[len++]=get_dights((int)temp);
			if(rest_dight_sum==0)
				break;
			memcpy(target,rest,sizeof(unsigned int)*real_length);
		}
		int start_pos=0;
		if(!sign)
			start_pos=1;
		for(int i=start_pos;i<(len+1)/2;++i)
		{
			char x=str[len-i-1+start_pos];
			str[len-i-1+start_pos]=str[i];
			str[i]=x;
		}
		str[len++]='\0';
	}
	//
	__host__ void print(int radix=10)
	{
		char str[bit_length];
		to_string(str,blocks,sign,radix,real_length);
		printf("%d------>%s\n",real_length,str);
	}
	__host__ __device__ void print_blocks() const
	{
		if(!sign)
			printf("- ");
		for(int i=0;i<real_length;++i)
			printf("%u,",this->blocks[i]);
		printf("\n");
	}

	__device__ __host__ unsigned char get_dight_num(char x)
	{
		if(x>='0'&&x<='9')
			return x-'0';
		else if(x>='A'&&x<='Z')
			return x-'A'+10;
		else
			return x-'a'+10;
	}

	__device__ __host__ int cuda_strlen(const char* str)
	{
		int len=0;
		if(str==NULL)
			return len;
		while(str[len])
			len++;
		return len;
	}

	__host__ __device__ int string_init(const char* n,int blocks[],bool& t_sign,const int radix)
	{
		int len=cuda_strlen(n);
		unsigned char target[bit_length];
		unsigned char rest[bit_length];
		int start_pos=0;
		if(n[0]=='-')
		{
			start_pos=1;
			t_sign=false;
		}
		else
			t_sign=true;
		for(int i=start_pos;i<len;++i)
			target[i-start_pos]=get_dight_num(n[i]);
		int block_id=0;
		while(true)
		{
			unsigned long long temp=0;
			unsigned long long rest_dight_sum=0;
			for(int i=0;i<len;++i)
			{
				temp=temp*radix+target[i];
				rest[i]=(int)(temp>>BLOCK_BIT_SIZE);
				temp=temp&BLOCK_MASK;
				rest_dight_sum=rest_dight_sum+rest[i];
			}
			blocks[block_id++]=(int)temp;
			if(rest_dight_sum==0)
				break;
			memcpy(target,rest,sizeof(char)*len);
		}
		return block_id;
	}

	__device__ __host__ int int_init(const long long n,int blocks[],bool& t_sign)
	{
		if(n<0)
			t_sign=false;
		else
			t_sign=true;
		long long extend_n=abs(n);
		int block_id=0;
		while(extend_n)
		{
			blocks[block_id]=(int)(extend_n&BLOCK_MASK);
			extend_n=extend_n>>BLOCK_BIT_SIZE;
			block_id++;
		}
		return block_id;
	}

	__device__ __host__ CUDA_GPUBigInt(const int n)
	{
		this->real_length=int_init(n,blocks,sign);
	}

	__device__ __host__ CUDA_GPUBigInt(const long long n)
	{
		this->real_length=int_init(n,blocks,sign);
	}

	__device__ __host__ CUDA_GPUBigInt()
	{
		this->sign=true;
		this->real_length=0;
	}

	__device__ __host__ void clear()
	{
		this->sign=true;
		this->real_length=0;
	}

	__device__ __host__ CUDA_GPUBigInt(const int blocks[],const int real_length,const bool sign)
	{
		this->sign=sign;
		this->real_length=real_length;
		memcpy(this->blocks,blocks,sizeof(int)*real_length);
	}

	__device__ __host__ CUDA_GPUBigInt(const CUDA_GPUBigInt& n)
	{
		memcpy(this->blocks,n.blocks,sizeof(int)*n.real_length);
		this->sign=n.sign;
		this->real_length=n.real_length;
	}

	__host__ __device__ CUDA_GPUBigInt(const char* n,int radix=10)
	{
		this->real_length=string_init(n,blocks,sign,radix);
	}

//对结构体CUDA_GPUBigInt结构的运算符重载
	//对struct CUDA_GPUBigInt等号重载
	__device__ __host__ CUDA_GPUBigInt& operator=(const CUDA_GPUBigInt& other)
	{
    	if (this != &other) {
        	this->sign = other.sign;
        	this->real_length = other.real_length;
        	for (int i = 0; i < other.real_length; ++i) {
            	this->blocks[i] = other.blocks[i];
        	}
        	// 若左侧原本更长，需将多余部分清零
        	for (int i = other.real_length; i < block_length; ++i) {
            	this->blocks[i] = 0;
        	}
    	}
    	return *this;
	}
	//对struct CUDA_GPUBigInt结构加号重载
	__device__ __host__ CUDA_GPUBigInt operator+(const CUDA_GPUBigInt& other) const
	{
    	CUDA_GPUBigInt result;
    	// 取较大长度
    	int max_len = (real_length > other.real_length) ? real_length : other.real_length;
    	result.real_length = max_len;
    	result.sign = true; // 这里只处理正数加法，若需处理符号可扩展

    	int carry = 0;
    	for (int i = 0; i < max_len; ++i) {
        	long long x = (i < real_length) ? (unsigned int)blocks[i] : 0;
        	long long y = (i < other.real_length) ? (unsigned int)other.blocks[i] : 0;
        	long long sum = x + y + carry;
        	result.blocks[i] = (int)(sum & BLOCK_MASK);
        	carry = (int)(sum >> BLOCK_BIT_SIZE);
    	}
    	if (carry != 0) {
        	result.blocks[max_len] = carry;
        	result.real_length = max_len + 1;
    	}
    	// 更新实际有效长度
    	result.real_length = get_real_length(result.blocks, result.real_length);
    	return result;
	}	
	//对struct CUDA_GPUBigInt结构 减号重载
	__device__ __host__ CUDA_GPUBigInt operator-(const CUDA_GPUBigInt& other) const
	{
    	CUDA_GPUBigInt result;
    	// 这里只处理正数减正数且this >= other，若需支持负数和更复杂情况请扩展
    	int borrow = 0;
    	int max_len = (real_length > other.real_length) ? real_length : other.real_length;
    	for (int i = 0; i < max_len; ++i) {
        	long long x = (i < real_length) ? (unsigned int)blocks[i] : 0;
        	long long y = (i < other.real_length) ? (unsigned int)other.blocks[i] : 0;
        	long long diff = x - y - borrow;
        	if (diff < 0) {
            	diff += BLOCK_MAX;
            	borrow = 1;
        	} else {
            	borrow = 0;
        	}
        	result.blocks[i] = (int)(diff & BLOCK_MASK);
    	}
    	result.real_length = get_real_length(result.blocks, max_len);
    	result.sign = true; // 这里只处理正数结果
    	return result;
	}

	//对struct CUDA_GPUBigInt结构 乘号重载
	__device__ __host__ CUDA_GPUBigInt operator*(const CUDA_GPUBigInt& other) const
	{
    	CUDA_GPUBigInt result;
    	result.real_length = real_length + other.real_length;
    	for (int i = 0; i < result.real_length; ++i) result.blocks[i] = 0;
    		for (int i = 0; i < real_length; ++i) {
        	long long carry = 0;
        	for (int j = 0; j < other.real_length; ++j) {
            	int k = i + j;
            	long long mul = (long long)(unsigned int)blocks[i] * (unsigned int)other.blocks[j] + result.blocks[k] + carry;
            	result.blocks[k] = (int)(mul & BLOCK_MASK);
            	carry = mul >> BLOCK_BIT_SIZE;
        	}
        	result.blocks[i + other.real_length] = (int)carry;
    	}
    	result.real_length = get_real_length(result.blocks, result.real_length);
    	result.sign = (sign == other.sign); // 异号为负，同号为正
    	return result;
	}

	//对struct CUDA_GPUBigInt结构 除号重载（仅支持单精度除法，假设other为int且不为0，适合演示和简单用途）
	__device__ __host__ CUDA_GPUBigInt operator/(const CUDA_GPUBigInt& other) const
	{
	    CUDA_GPUBigInt result;
	    CUDA_GPUBigInt remainder;
	    result.real_length = real_length;
	    remainder.real_length = real_length;
	    for (int i = 0; i < real_length; ++i) {
    	    remainder.blocks[i] = blocks[i];
    	}
    	// 这里只实现简单的短除法（other为单个int），如需支持大整数除法请用专用算法
    	if (other.real_length == 1) {
        	unsigned long long divisor = (unsigned int)other.blocks[0];
        	unsigned long long rem = 0;
        	for (int i = real_length - 1; i >= 0; --i) {
            	unsigned long long cur = (rem << BLOCK_BIT_SIZE) | (unsigned int)blocks[i];
            	result.blocks[i] = (int)(cur / divisor);
            	rem = cur % divisor;
        	}
        	result.real_length = get_real_length(result.blocks, real_length);
        	result.sign = (sign == other.sign);
        	return result;
    	} else {
        	// 若需支持大整数除法，请调用已有的divide函数或实现长除法
        	// 这里只返回0作为占位
        	result.real_length = 1;
        	result.blocks[0] = 0;
        	result.sign = true;
        	return result;
    	}
	}	
};
//-----------------------------------------------------------------------------------------------
__device__ __host__ void bigint_right_shift(const CUDA_GPUBigInt& target,CUDA_GPUBigInt& result,int bit)
{
	int shift_block=bit/BLOCK_BIT_SIZE;
	int rest_shift=bit%BLOCK_BIT_SIZE;
	int target_len=target.real_length;
	unsigned int register_c[block_length];
	for(int i=shift_block;i<target_len;++i)
		register_c[i-shift_block]=(unsigned int)target.blocks[i];
	unsigned int low_mask=(1<<rest_shift)-1;
	for(int i=0;i<target_len-shift_block-1;++i)
		register_c[i]=((register_c[i+1]&low_mask)<<(BLOCK_BIT_SIZE-rest_shift))|(register_c[i]>>rest_shift);
	register_c[target_len-shift_block-1]=register_c[target_len-shift_block-1]>>rest_shift;
	for(int i=0;i<target_len-shift_block;++i)
		result.blocks[i]=(int)register_c[i];
	result.real_length=get_real_length(register_c,target_len-shift_block);
	result.sign=target.sign;
}

__device__ __host__ void bigint_left_shift(CUDA_GPUBigInt& target,CUDA_GPUBigInt& result,int bit)
{
	int shift_block=bit/BLOCK_BIT_SIZE;
	int rest_shift=bit%BLOCK_BIT_SIZE;
	int target_len=target.real_length;
	unsigned long long register_c[block_length];
	for(int i=0;i<shift_block;++i)
		register_c[i]=0;
	for(int i=shift_block;i<target_len+shift_block;++i)
		register_c[i]=(unsigned long long)(unsigned int)target.blocks[i-shift_block];
	unsigned long long k=0;
	for(int i=shift_block;i<target_len+shift_block;++i)
	{
		register_c[i]=(register_c[i]<<rest_shift)+k;
		k=register_c[i]>>BLOCK_BIT_SIZE;
		register_c[i]=register_c[i]&BLOCK_MASK;
	}
	for(int i=0;i<target_len+shift_block;++i)
		result.blocks[i]=(int)register_c[i];
	result.blocks[target_len+shift_block]=(int)k;
	result.sign=target.sign;
	result.real_length=get_real_length(result.blocks,target_len+shift_block+1);
}

__device__ __host__ inline int compare_abs(const int a[],const int a_len,const int b[],const int b_len)
{
	if(a_len>b_len)
		return 1;
	else if(a_len<b_len)
		return -1;
	else
	{
		for(int i=a_len-1;i>=0;i--)
		{
			long long x=(long long)(unsigned int)a[i];
			long long y=(long long)(unsigned int)b[i];
			if(x>y)
				return 1;
			else if(x==y)
				continue;
			else
				return -1;
		}
		return 0;
	}
}

__device__ __host__ inline bool is_bigger_equal(const CUDA_GPUBigInt& a,const CUDA_GPUBigInt& b)
{
	if((a.sign)&&(!b.sign))
		return true;
	else if((!a.sign)&&(b.sign))
		return false;
	else
	{
		int r=compare_abs(a.blocks,a.real_length,b.blocks,b.real_length);
		if(a.sign&&b.sign)
			return r>=0;
		else
			return r<=0;
	}
}

__device__ __host__ inline bool is_bigger(const CUDA_GPUBigInt& a,const CUDA_GPUBigInt& b)
{
	if((a.sign)&&(!b.sign))
		return true;
	else if((!a.sign)&&(b.sign))
		return false;
	else
	{
		int r=compare_abs(a.blocks,a.real_length,b.blocks,b.real_length);
		if(a.sign&&b.sign)
			return r>0;
		else
			return r<0;
	}
}

__device__ __host__ inline bool is_smaller_equal(const CUDA_GPUBigInt& a,const CUDA_GPUBigInt& b)
{
	if((a.sign)&&(!b.sign))
		return false;
	else if((!a.sign)&&(b.sign))
		return true;
	else
	{
		int r=compare_abs(a.blocks,a.real_length,b.blocks,b.real_length);
		if(a.sign&&b.sign)
			return r<=0;
		else
			return r>=0;
	}
}

__device__ __host__ inline bool is_smaller(const CUDA_GPUBigInt& a,const CUDA_GPUBigInt& b)
{

	if((a.sign)&&(!b.sign))
		return false;
	else if((!a.sign)&&(b.sign))
		return true;
	else
	{
		int r=compare_abs(a.blocks,a.real_length,b.blocks,b.real_length);
		if(a.sign&&b.sign)
			return r<0;
		else
			return r>0;
	}
}

__device__ __host__ inline bool is_equal(const CUDA_GPUBigInt& a,const CUDA_GPUBigInt& b)
{
	if(a.sign!=b.sign)
		return false;
	if(a.real_length!=b.real_length)
		return false;
	for(int i=0;i<a.real_length;++i)
	{
		long long x=(long long)(unsigned int)a.blocks[i];
		long long y=(long long)(unsigned int)b.blocks[i];
		if(x!=y)
			return false;
	}
	return true;
}


__device__ __host__ int count_leading_zeros(int x)
{
#ifdef __CUDA_ARCH__
    return __clz(x);
#else
    return __builtin_clz(x); // 替换__lzcnt为__builtin_clz
#endif
}

__device__ __host__ void tiny_mult(const CUDA_GPUBigInt& a,unsigned int b,CUDA_GPUBigInt& c)
{
	int a_len=a.real_length;
	if(a_len==0||b==0)
	{
		c.real_length=0;
		return;
	}
	unsigned long long register_c[block_length];
	int total_len=a_len+1;
	memset(register_c,0,sizeof(unsigned long long)*(total_len));
	unsigned long long x,y;
	for(int i=0;i<a_len;++i)
	{
		x=(unsigned long long)(unsigned int)a.blocks[i];
		y=(unsigned long long)(unsigned int)b;
		register_c[i]=register_c[i]+x*y;
		register_c[i+1]=register_c[i+1]+(register_c[i]>>BLOCK_BIT_SIZE);
		register_c[i]=register_c[i]&BLOCK_MASK;
	}
	for(int i=0;i<total_len;++i)
	{
		register_c[i+1]=register_c[i+1]+(register_c[i]>>BLOCK_BIT_SIZE);
		c.blocks[i]=(int)(register_c[i]&BLOCK_MASK);
	}
	c.real_length=get_real_length(register_c,total_len);
	c.sign=a.sign;
}
// 功能说明：
// 该函数实现了两个CUDA_GPUBigInt类型大整数的加法运算，支持在CUDA设备端（GPU）和主机端（CPU）调用。
// 实现思路为：逐位相加，处理进位，结果写入c，自动处理不同长度。
// 具体流程：
// 1. 先对b的每一位与a对应位相加，并加上进位，结果写入c。
// 2. 如果a比b长，则继续处理a剩余的高位，并加上进位。
// 3. 最后将最终进位写入c的最高位，并更新c的实际长度。
__device__ __host__ void bigint_add(const CUDA_GPUBigInt& a,const CUDA_GPUBigInt& b,CUDA_GPUBigInt& c)
{
	int carry=0;
	for(int i=0;i<b.real_length;++i)
	{
		long long x=(long long)(unsigned int)a.blocks[i];
		long long y=(long long)(unsigned int)b.blocks[i];
		long long block_sum=x+y+carry;
		carry=(int)(block_sum>>BLOCK_BIT_SIZE);
		c.blocks[i]=(int)(block_sum&BLOCK_MASK);
	}
	for(int i=b.real_length;i<a.real_length;++i)
	{
		if(carry==0)
		{
			c.blocks[i]=a.blocks[i];
			continue;
		}
		long long x=(long long)(unsigned int)a.blocks[i];
		long long block_sum=x+carry;
		carry=(int)(block_sum>>BLOCK_BIT_SIZE);
		c.blocks[i]=(int)(block_sum&BLOCK_MASK);
	}
	c.blocks[a.real_length]=carry;
	c.real_length=get_real_length(c.blocks,a.real_length+1);
}
/*
实现了大整数与32位整数的加法运算，支持在CUDA设备端（GPU）和主机端（CPU）调用。
其主要功能是：计算大整数a与32位整数b的和，并将结果存入c。实现思路为将b加到最低位，逐位处理进位，最后根据实际有效位数修正结果长度。
*/
__device__ __host__ void bigint_add(const CUDA_GPUBigInt& a, const int b, CUDA_GPUBigInt& c)
{
    int carry = 0;
    // 先加最低位
    if (a.real_length > 0) {
        long long x = (unsigned int)a.blocks[0];
        long long block_sum = x + (unsigned int)b;
        c.blocks[0] = (int)(block_sum & BLOCK_MASK);
        carry = (int)(block_sum >> BLOCK_BIT_SIZE);
    } else {
        c.blocks[0] = b;
        carry = 0;
    }

    // 处理剩余位
    int i = 1;
    for (; i < a.real_length; ++i) {
        if (carry == 0) {
            c.blocks[i] = a.blocks[i];
        } else {
            long long x = (unsigned int)a.blocks[i];
            long long block_sum = x + carry;
            c.blocks[i] = (int)(block_sum & BLOCK_MASK);
            carry = (int)(block_sum >> BLOCK_BIT_SIZE);
        }
    }
    // 如果还有进位
    if (carry != 0) {
        c.blocks[i++] = carry;
    }
    c.real_length = get_real_length(c.blocks, (a.real_length > i ? a.real_length : i));
    c.sign = a.sign;
}
/*
实现了大整数与64位整数的加法运算，支持在CUDA设备端（GPU）和主机端（CPU）调用。
其主要功能是：计算大整数a与64位整数b的和，并将结果存入c。
实现思路为将b拆分为低32位和高32位，逐位相加并处理进位，最后根据实际有效位数修正结果长度。
*/
__device__ __host__ void bigint_add(const CUDA_GPUBigInt& a, const int64_t b, CUDA_GPUBigInt& c)
{
    int carry = 0;
    uint32_t b_low = (uint32_t)(b & 0xFFFFFFFFULL);
    uint32_t b_high = (uint32_t)((b >> 32) & 0xFFFFFFFFULL);

    // 先加低32位
    if (a.real_length > 0) {
        long long x = (unsigned int)a.blocks[0];
        long long block_sum = x + b_low;
        c.blocks[0] = (int)(block_sum & BLOCK_MASK);
        carry = (int)(block_sum >> BLOCK_BIT_SIZE);
    } else {
        c.blocks[0] = b_low;
        carry = 0;
    }

    // 加高32位
    if (a.real_length > 1) {
        long long x = (unsigned int)a.blocks[1];
        long long block_sum = x + b_high + carry;
        c.blocks[1] = (int)(block_sum & BLOCK_MASK);
        carry = (int)(block_sum >> BLOCK_BIT_SIZE);
    } else {
        long long block_sum = b_high + carry;
        c.blocks[1] = (int)(block_sum & BLOCK_MASK);
        carry = (int)(block_sum >> BLOCK_BIT_SIZE);
    }

    // 处理剩余位
    int i = 2;
    for (; i < a.real_length; ++i) {
        if (carry == 0) {
            c.blocks[i] = a.blocks[i];
        } else {
            long long x = (unsigned int)a.blocks[i];
            long long block_sum = x + carry;
            c.blocks[i] = (int)(block_sum & BLOCK_MASK);
            carry = (int)(block_sum >> BLOCK_BIT_SIZE);
        }
    }
    // 如果还有进位
    if (carry != 0) {
        c.blocks[i++] = carry;
    }
    c.real_length = get_real_length(c.blocks, (a.real_length > i ? a.real_length : i));
    c.sign = a.sign;
}

//两个数相减
__device__ __host__ void bigint_subtract(const CUDA_GPUBigInt& a,const CUDA_GPUBigInt& b,CUDA_GPUBigInt& c)
{
    // 判断 a 和 b 的大小
    if (is_bigger_equal(a, b)) {
        // a >= b，结果为正
        int carry = 0;
        for (int i = 0; i < b.real_length; ++i) {
            long long x = (long long)(unsigned int)a.blocks[i];
            long long y = (long long)(unsigned int)b.blocks[i];
            long long block_sum = x - y - carry;
            carry = 0;
            if (block_sum < 0)
                carry = 1;
            c.blocks[i] = (int)((block_sum + BLOCK_MAX) & BLOCK_MASK);
        }
        for (int i = b.real_length; i < a.real_length; ++i) {
            if (carry == 0) {
                c.blocks[i] = a.blocks[i];
                continue;
            }
            long long x = (long long)(unsigned int)a.blocks[i];
            long long block_sum = x - carry;
            carry = 0;
            if (block_sum < 0)
                carry = 1;
            c.blocks[i] = (int)((block_sum + BLOCK_MAX) & BLOCK_MASK);
        }
        c.real_length = get_real_length(c.blocks, a.real_length);
        c.sign = true;
    } else {
        // a < b，结果为负，计算 b - a
        int carry = 0;
        for (int i = 0; i < a.real_length; ++i) {
            long long x = (long long)(unsigned int)b.blocks[i];
            long long y = (long long)(unsigned int)a.blocks[i];
            long long block_sum = x - y - carry;
            carry = 0;
            if (block_sum < 0)
                carry = 1;
            c.blocks[i] = (int)((block_sum + BLOCK_MAX) & BLOCK_MASK);
        }
        for (int i = a.real_length; i < b.real_length; ++i) {
            if (carry == 0) {
                c.blocks[i] = b.blocks[i];
                continue;
            }
            long long x = (long long)(unsigned int)b.blocks[i];
            long long block_sum = x - carry;
            carry = 0;
            if (block_sum < 0)
                carry = 1;
            c.blocks[i] = (int)((block_sum + BLOCK_MAX) & BLOCK_MASK);
        }
        c.real_length = get_real_length(c.blocks, b.real_length);
        c.sign = false;
    }
    // 如果结果为0，sign应为正
    if (c.real_length == 0) {
        c.sign = true;
    }
}
//两个数相减(减一个int_32整数)
__device__ __host__ void bigint_subtract(const CUDA_GPUBigInt& a,unsigned int& b,CUDA_GPUBigInt& c)
{
	int carry=0;
	long long x=(long long)(unsigned int)a.blocks[0];
	long long y=(long long)(unsigned int)b;
	long long block_sum=x-y-carry;
	if(block_sum<0)
		carry=1;
	c.blocks[0]=(int)((block_sum+BLOCK_MAX)&BLOCK_MASK);

	for(int i=1;i<a.real_length;++i)
	{
		if(carry==0)
		{
			c.blocks[i]=a.blocks[i];
			continue;
		}
		long long x=(long long)(unsigned int)a.blocks[i];
		long long block_sum=x-carry;
		carry=0;
		if(block_sum<0)
			carry=1;
		c.blocks[i]=(int)((block_sum+BLOCK_MAX)&BLOCK_MASK);
	}
	c.real_length=get_real_length(c.blocks,a.real_length);
}

/*
实现了两个大整数的高精度乘法运算，支持在CUDA设备端（GPU）和主机端（CPU）调用。  
其主要功能是：计算大整数a和b的乘积，并将结果存入c。实现思路为模拟手工乘法，
逐位相乘并累加到结果数组中，处理进位，最后根据实际有效位数修正结果长度。
*/
__device__ __host__ void bigint_mult(const CUDA_GPUBigInt& a,const CUDA_GPUBigInt& b,CUDA_GPUBigInt& c)
{
	int a_len=a.real_length;
	int b_len=b.real_length;
	if(a_len==0||b_len==0)
	{
		c.real_length=0;
		return;
	}
	unsigned long long register_c[block_length];
	int total_len=a_len+b_len;
	memset(register_c,0,sizeof(unsigned long long)*(total_len));
	for(int i=0;i<a_len;++i)
	{
		for(int j=0;j<b_len;++j)
		{
			unsigned long long x=(unsigned long long)(unsigned int)a.blocks[i];
			unsigned long long y=(unsigned long long)(unsigned int)b.blocks[j];
			register_c[i+j]=register_c[i+j]+x*y;
			register_c[i+j+1]=register_c[i+j+1]+(register_c[i+j]>>BLOCK_BIT_SIZE);
			register_c[i+j]=register_c[i+j]&BLOCK_MASK;
		}
	}
	for(int i=0;i<total_len;++i)
	{
		register_c[i+1]=register_c[i+1]+(register_c[i]>>BLOCK_BIT_SIZE);
		c.blocks[i]=(int)(register_c[i]&BLOCK_MASK);
	}
	c.real_length=get_real_length(register_c,total_len);
	c.sign = (a.sign == b.sign); //赋值正负符号
}
/*
实现了大整数与32位无符号整数的高精度乘法运算，支持在CUDA设备端（GPU）和主机端（CPU）调用。
其主要功能是：计算大整数a与32位整数b的乘积，并将结果存入c。实现思路为逐位相乘并累加到结果数组中，处理进位，最后根据实际有效位数修正结果长度。
*/
__device__ __host__ void bigint_mult(const CUDA_GPUBigInt& a, unsigned int b, CUDA_GPUBigInt& c)
{
    int a_len = a.real_length;
    if (a_len == 0 || b == 0)
    {
        c.real_length = 0;
        return;
    }
    unsigned long long register_c[block_length] = {0};
    int total_len = a_len + 1;
    for (int i = 0; i < a_len; ++i)
    {
        unsigned long long x = (unsigned long long)(unsigned int)a.blocks[i];
        unsigned long long y = (unsigned long long)b;
        register_c[i] += x * y;
        register_c[i + 1] += (register_c[i] >> BLOCK_BIT_SIZE);
        register_c[i] = register_c[i] & BLOCK_MASK;
    }
    for (int i = 0; i < total_len; ++i)
    {
        register_c[i + 1] += (register_c[i] >> BLOCK_BIT_SIZE);
        c.blocks[i] = (int)(register_c[i] & BLOCK_MASK);
    }
    c.real_length = get_real_length(register_c, total_len);
    c.sign = a.sign;
}
/*
实现了大整数与64位整数的高精度乘法运算，支持在CUDA设备端（GPU）和主机端（CPU）调用。
其主要功能是：计算大整数a与64位整数b的乘积，并将结果存入c。实现思路为将b拆分为低32位和高32位，分别与a相乘，结果累加并处理进位，最后根据实际有效位数修正结果长度。
*/
__device__ __host__ void bigint_mult(const CUDA_GPUBigInt& a, int64_t b, CUDA_GPUBigInt& c)
{
    unsigned int b_low = (unsigned int)(b & 0xFFFFFFFFULL);
    unsigned int b_high = (unsigned int)((b >> 32) & 0xFFFFFFFFULL);

    CUDA_GPUBigInt temp_low, temp_high;

    // a * b_low
    bigint_mult(a, b_low, temp_low);

    // a * b_high，结果左移32位
    if (b_high != 0) {
        bigint_mult(a, b_high, temp_high);
        // 左移32位（即一个block）
        for (int i = temp_high.real_length; i > 0; --i) {
            temp_high.blocks[i] = temp_high.blocks[i - 1];
        }
        temp_high.blocks[0] = 0;
        temp_high.real_length += 1;
    } else {
        temp_high.real_length = 0;
    }

    // temp_low + temp_high
    bigint_add(temp_low, temp_high, c);
    c.sign = a.sign;
}

__device__ __host__ CUDA_GPUBigInt knuth_div(CUDA_GPUBigInt& a,CUDA_GPUBigInt& b)
{
	int b_len=b.real_length;
	int move_block=1;
	int d=count_leading_zeros(b.blocks[b_len-1])+BLOCK_BIT_SIZE*move_block;
	bigint_left_shift(a,a,d);
	bigint_left_shift(b,b,d);
	int u_len=a.real_length;
	int v_len=b.real_length;
	unsigned long long v1=(unsigned long long)(unsigned int)b.blocks[v_len-1];
	unsigned long long v2=(unsigned long long)(unsigned int)b.blocks[v_len-2];
	a.blocks[u_len]=0,u_len=u_len+1;
	a.real_length=u_len;
	int ret_len=u_len-v_len;
	int f_len=v_len+1;
	int head_pos=ret_len-1;
	unsigned long long u0,u1,u2,chunk,q_hat,r_hat;
	CUDA_GPUBigInt register_f,register_k,register_ret;
	for(int i=head_pos+1;i<head_pos+f_len;++i)
		register_f.blocks[i-head_pos]=a.blocks[i];
	register_f.real_length=f_len;
	register_f.sign=true;
	unsigned int tiny;
	for(int i=0;i<ret_len;++i)
	{
		int head_pos=ret_len-i-1;
		register_f.blocks[0]=a.blocks[head_pos];
		register_f.real_length=get_real_length(register_f.blocks,f_len);
		u0=(unsigned long long)(unsigned int)register_f.blocks[f_len-1];
		u1=(unsigned long long)(unsigned int)register_f.blocks[f_len-2];
		u2=(unsigned long long)(unsigned int)register_f.blocks[f_len-3];
		chunk=(u0<<BLOCK_BIT_SIZE)|u1;
		q_hat=chunk/v1;
		r_hat=chunk-q_hat*v1;
		if(q_hat*v2>((r_hat<<BLOCK_BIT_SIZE)|u2))
			q_hat=q_hat-1;
		tiny=(unsigned int)(q_hat&BLOCK_MASK);
		tiny_mult(b,tiny,register_k);
		if(is_bigger(register_k,register_f))
		{
			q_hat=q_hat-1;
			tiny=(unsigned int)(q_hat&BLOCK_MASK);
			tiny_mult(b,tiny,register_k);
		}
		bigint_subtract(register_f,register_k,register_f);
		for(int j=f_len-1;j>=register_f.real_length;j--)
			register_f.blocks[j]=0;
		for(int j=f_len-1;j>=1;--j)
			register_f.blocks[j]=register_f.blocks[j-1];
		register_ret.blocks[head_pos]=(int)tiny;
	}
	bigint_right_shift(a,a,d);
	bigint_right_shift(b,b,d);
	register_ret.real_length=get_real_length(register_ret.blocks,ret_len);
	register_ret.sign=true;
	return register_ret;
}

/*
大整数的除法和取模（模运算）功能。  
具体来说，`bigint_div_mod` 函数根据 `is_mod` 参数，决定是执行除法还是取模操作：
- 当 `is_mod` 为 `false` 时，计算 `a / b`，结果存入 `c`，即大整数除法。
- 当 `is_mod` 为 `true` 时，计算 `a % b`，结果存入 `c`，即大整数取模。
实现细节如下：
- 如果 `a < b`，则除法结果为 0，模运算结果为 `a` 本身。
- 否则，调用高精度除法 `knuth_div` 计算商（除法结果），并根据 `is_mod` 决定返回商还是余数。
- 余数的计算方式为：`余数 = a - b * 商`。
该函数支持在 CUDA 设备端和主机端调用，适用于高性能并行环境下的大整数除法和模运算。
*/
__device__ __host__ void bigint_div_mod(const CUDA_GPUBigInt& a,const CUDA_GPUBigInt& b,CUDA_GPUBigInt& c,bool is_mod)
{
	if(is_smaller(a,b))
	{
		if(is_mod)
		{
			int a_len=a.real_length;
			for(int i=0;i<a_len;++i)
				c.blocks[i]=a.blocks[i];
			c.real_length=a_len;
			c.sign=true;
		}
		else
		{
			c.real_length=0;
		}
		return;
	}
	if(!is_mod){
		CUDA_GPUBigInt ta(a), tb(b);
		c=knuth_div(ta,tb);
	}
	else
	{
		CUDA_GPUBigInt ta(a), tb(b);
		CUDA_GPUBigInt register_r=knuth_div(ta,tb);
		bigint_mult(b,register_r,register_r);
		bigint_subtract(a,register_r,c);
	}
}
//计算a%b的结果存入c
__device__ __host__ void bigint_mod(const CUDA_GPUBigInt& a,const CUDA_GPUBigInt& b, CUDA_GPUBigInt& c)
{
	if (is_smaller(a, b)) {
		int a_len = a.real_length;
		for (int i = 0; i < a_len; ++i)
			c.blocks[i] = a.blocks[i];
		c.real_length = a_len;
		c.sign = true;
		return;
	}

    // 用临时变量保护 a, b
    CUDA_GPUBigInt ta(a), tb(b);
    CUDA_GPUBigInt quotient = knuth_div(ta, tb); // ta, tb 被修改没关系
    CUDA_GPUBigInt prod;
    bigint_mult(b, quotient, prod); // 注意这里用原始 b
    bigint_subtract(a, prod, c);    // 用原始 a
    // 保证余数非负
    if (!c.sign && c.real_length > 0) {
        // c = c + b
        bigint_add(c, b, c);
        c.sign = true;
    }
    c.real_length = get_real_length(c.blocks, block_length);


}

/*
实现大整数的模幂运算（即 c = a^b mod m），支持在CUDA设备端和主机端调用。  
其功能是：计算大整数a的b次幂再对m取模，结果存入c。实现采用了“快速幂”算法，
每次判断指数b的最低位，如果为1则将当前结果与底数相乘并取模，然后底数自乘并取模，
指数右移一位，循环直到指数为0。
*/
__device__ __host__ void bigint_power_mod(const CUDA_GPUBigInt& a,const CUDA_GPUBigInt& b,const CUDA_GPUBigInt& m,CUDA_GPUBigInt& c)
{
	CUDA_GPUBigInt times(b);
	CUDA_GPUBigInt temp(a);
	c.blocks[0]=1;c.sign=true;c.real_length=1;
	while(times.real_length>0)
	{
		if(times.blocks[0]&1)
		{
			bigint_mult(c,temp,c);
			bigint_div_mod(c,m,c,true);
		}
		bigint_mult(temp,temp,temp);
		bigint_div_mod(temp,m,temp,true);
		bigint_right_shift(times,times,1);
	}
}
/*
实现大整数的模幂运算（即 c = a^b mod m），支持在CUDA设备端和主机端调用。  
其功能是：计算大整数a的b次幂再对m取模，结果存入c。实现采用了“快速幂”算法，
每次判断指数b的最低位，如果为1则将当前结果与底数相乘并取模，然后底数自乘并取模，
指数右移一位，循环直到指数为0。
这里传入的幂指数b是一个指针
*/
__device__ __host__ void bigint_power_mod(const CUDA_GPUBigInt& a,const CUDA_GPUBigInt* b,const CUDA_GPUBigInt& m,CUDA_GPUBigInt& c)
{
	CUDA_GPUBigInt times(*b);
	CUDA_GPUBigInt temp(a);
	c.blocks[0]=1;c.sign=true;c.real_length=1;
	while(times.real_length>0)
	{
		if(times.blocks[0]&1)
		{
			bigint_mult(c,temp,c);
			bigint_div_mod(c,m,c,true);
		}
		bigint_mult(temp,temp,temp);
		bigint_div_mod(temp,m,temp,true);
		bigint_right_shift(times,times,1);
	}
}
/*
实现大整数的模幂运算（即 c = a^b mod m），支持在CUDA设备端和主机端调用。  
其功能是：计算大整数a的b次幂再对m取模，结果存入c。实现采用了“快速幂”算法，
每次判断指数b的最低位，如果为1则将当前结果与底数相乘并取模，然后底数自乘并取模，
指数右移一位，循环直到指数为0。
这里传入的幂指数b及模数m是一个指针
*/
__device__ __host__ void bigint_power_mod(const CUDA_GPUBigInt& a,const CUDA_GPUBigInt* b,const CUDA_GPUBigInt* m,CUDA_GPUBigInt& c)
{
	CUDA_GPUBigInt times(*b),mod(*m);
	CUDA_GPUBigInt temp(a);
	c.blocks[0]=1;c.sign=true;c.real_length=1;
	while(times.real_length>0)
	{
		if(times.blocks[0]&1)
		{
			bigint_mult(c,temp,c);
			bigint_div_mod(c,mod,c,true);
		}
		bigint_mult(temp,temp,temp);
		bigint_div_mod(temp,mod,temp,true);
		bigint_right_shift(times,times,1);
	}
}
/*
计算a相对于模m的乘法逆元,即ret=a^(m-2) mod m。
这是基于费马小定理的一个应用，前提是m是素数。
*/
__device__ __host__ void bigint_invert(const CUDA_GPUBigInt& a,const CUDA_GPUBigInt& m,CUDA_GPUBigInt& ret)
{
	CUDA_GPUBigInt power;
	unsigned int subv=2;
	bigint_subtract(m,subv,power);
	bigint_power_mod(a,&power,&m,ret);
}
//-----------------------------------------------------------------------------------------------
// GPUBigInt类，封装常用的大整数GPU运算
class GPUBigInt {
public:
    CUDA_GPUBigInt value;

    // 构造函数，传入大整数的int_size，初始化CUDA_GPUBigInt
	// 默认int_size为8，即大整数使用8个32位整数表示
    __device__ __host__ GPUBigInt(int int_size = 8) {
        value.real_length = int_size;
        // 初始化blocks为0
        for (int i = 0; i < int_size; ++i) {
            value.blocks[i] = 0;
        }
        value.sign = 0;
    }
    // 通过十六进制字符串初始化GPUBigInt
	__device__ __host__ GPUBigInt(const char* num_str,const int radix = 10) {
        // 正确调用成员函数
		value.sign = true;
		if(num_str[0]=='-')
			value.sign = false;
        value.real_length = value.string_init(num_str, value.blocks, value.sign, radix);
	}
public:	//GPUBigInt类的功能函数
    // 类内大整数与一个传入的大整数相加
	__device__ __host__ void add(const CUDA_GPUBigInt& to_add, CUDA_GPUBigInt& add_ret) {
		bigint_add(value, to_add, add_ret);
	}
    // 类内大整数与一个传入的大整数相减,to_sub是被减数，sub_ret是结果
    __device__ __host__ void sub(const CUDA_GPUBigInt& to_sub, CUDA_GPUBigInt& sub_ret) {
        bigint_subtract(value, to_sub, sub_ret);
    }

    // 类内大整数与一个传入的大整数相乘
    __device__ __host__ void mul(const CUDA_GPUBigInt& to_mul, CUDA_GPUBigInt& mul_ret) {
        bigint_mult(value, to_mul, mul_ret);
    }
    // 类内大整数与一个传入的大整数相除
    __device__ __host__ void div(const CUDA_GPUBigInt& to_div, CUDA_GPUBigInt& div_ret) {
        bigint_div_mod(value, to_div, div_ret, false);
    }
    // 类内大整数取模运算
    __device__ __host__ void mod(const  CUDA_GPUBigInt& modulus, CUDA_GPUBigInt& mod_ret) {
        bigint_mod(value, modulus, mod_ret);
    }
    // 类内大整数幂模运算
    __device__ __host__ void powmod(const CUDA_GPUBigInt& exponent,const  CUDA_GPUBigInt& modulus, CUDA_GPUBigInt& powmod_ret) {
        bigint_power_mod(value, exponent, modulus, powmod_ret);
    }
    //计算乘法逆元
    __device__ __host__ void invert(const CUDA_GPUBigInt& modulus, CUDA_GPUBigInt& inv_ret) {
        bigint_invert(value, modulus, inv_ret);
    }
	//转换为十进制字符串1
	__device__ __host__ void to_char(char* ret,int radix = 10)  const{
		CUDA_GPUBigInt tv(value);
		tv.to_str(ret,value.blocks,value.real_length,value.sign,radix);
	}
    // 返回十进制字符串
    __host__ std::string to_string(int radix = 10)  const{
        char str[bit_length] = {0};
        to_char(str,radix);
        return std::string(str);
    }

public:		//重载运算符
	// 加号重载
	__device__ __host__ GPUBigInt operator+(const GPUBigInt& other) const {
    	GPUBigInt result; // 默认构造，分配最大 block_length
		bigint_add(value,other.value, result.value);
    	return result;
	}
	//	等号重载
	__device__ __host__ GPUBigInt& operator=( GPUBigInt& other) {
    	if (this != &other) {
        	value.real_length = other.value.real_length;
        	value.sign = other.value.sign;
        	for (int i = 0; i < value.real_length; ++i) {
            	value.blocks[i] = other.value.blocks[i];
        	}
    	}
    	return *this;
	}	
	// 减号重载
	__device__ __host__ GPUBigInt operator-(const GPUBigInt& other) const {
    	GPUBigInt result; // 默认构造，分配最大 block_length
    	bigint_subtract(value, other.value, result.value);
    	return result;
	}
	// 乘号重载
	__device__ __host__ GPUBigInt operator*(const GPUBigInt& other) const {
    	GPUBigInt result; // 默认构造，分配最大 block_length
    	bigint_mult(value, other.value, result.value);
    	return result;
	}
	// 除号重载
	__device__ __host__ GPUBigInt operator/(const GPUBigInt& other) const {
    	GPUBigInt result; // 默认构造，分配最大 block_length
    	bigint_div_mod(value, other.value, result.value, false);
    	return result;
	}
	// 取模重载
	__device__ __host__ GPUBigInt operator%(const GPUBigInt& other) const {
    	GPUBigInt result; // 默认构造，分配最大 block_length
    	bigint_mod(value, other.value, result.value);
    	return result;
	}
	// 取模重载
	__device__ __host__ GPUBigInt operator==(const GPUBigInt& other) const {
    	return is_bigger_equal(value, other.value);
	}
};
//-----------------------------------------------------------------------------------------------