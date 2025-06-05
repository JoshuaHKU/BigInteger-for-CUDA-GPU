# cuda-bigint

## Project Overview

`cuda-bigint` A lightweight CUDA library for high-performance big integer arithmetic on GPUs.​​ 
Provides core operations (addition, subtraction, multiplication, division) and essential cryptographic 
primitives including modular exponentiation, multiplicative inverse (modular), and bitwise shifts.

## Main Features

- **High-precision big integer arithmetic**: Addition, subtraction, multiplication, division, modulo, comparison, bitwise operations—all supporting GPU parallel computation.
- **Modular exponentiation**: Supports big integer power-mod operations, suitable for cryptographic scenarios.
- **Multiplicative inverse**: Supports calculation of the multiplicative inverse of big integers.
- **Base conversion and string initialization**: Supports conversion between decimal, hexadecimal, and other bases, as well as initialization from strings.
- **Random big integer generation**: Supports bitwise random big integer generation, can be combined with GMP to generate large primes.
- **Unified host and device interface**: All core operations can be called on both CPU and GPU, making testing and integration convenient.

## Code Structure

- `cuda_bigint.h`  
  Core Implementation of Big Integer Data Structures and Algorithms​​.Includes CUDA_GPUBigInt struct, GPUBigInt class, and all related big integer arithmetic functions.
  ​​Usage​​: Simply add this file to your GPU project and include the header to use full functionality.
- `test_gpubigint.cu`  
  Functional tests for big integer operations on the GPU, including addition, subtraction, multiplication, division, modulo, multiplicative inverse, etc., to verify correctness and performance on CUDA devices.

## Quick Start

1. **Requirements**
   - CUDA 10.x or above
   - NVIDIA GPU with CUDA support
   - C++11 or above compiler

2. **Build**
   Tested on Ubuntu 20.
   ```bash
   make all
   ./test_gpubigint
   ```

Reference: https://github.com/bingshen/cuda-bigint

#--------------------------------------------------------------------------------------
# cuda-bigint

## 项目简介

`cuda-bigint` 是一个基于 CUDA 的高性能大整数运算库，支持在 GPU 上进行高精度整数的加、减、乘、除、取余、比较、高次幂取余、位移等操作。
项目目标是为加密算法（如 RSA、ECC 等）提供高效的可以GPU设备侧运行的大整数基础运算。

## 主要功能

- **高精度大整数运算**：加法、减法、乘法、除法、取模、比较、位移等，全部支持 GPU 并行计算。
- **高次幂模运算**：支持大整数的幂模（power-mod）运算，适用于加密算法中的模幂运算场景。
- **乘法逆元**：支持大整数的乘法逆元计算。
- **进制转换与字符串初始化**：支持十进制、十六进制等多进制字符串与大整数的相互转换。
- **随机大整数生成**：支持按位随机生成大整数，可结合 GMP 生成大质数。
- **主机端与设备端统一接口**：所有核心运算均可在 CPU 和 GPU 上调用，便于测试与集成。

## 代码结构

- `cuda_bigint.h`  
  大整数核心数据结构与算法实现，包含CUDA_GPUBigInt结构、GPUBigInt 类及所有大整数运算相关功能函数。把此文件直接加入GPU项目并引用此头文件即可直接使用。
- `test_gpubigint.cu`  
  对cuda_bigint.h内实现的GPUBingInt 类及相关大整数运算功能函数的测试，包括在GPU的设备上实现实例的创建、赋值、加减乘除、模运算、乘法逆元等典型用例。


## 快速开始

1. **环境要求**
   - CUDA 10.x 及以上
   - 支持 CUDA 的 NVIDIA GPU
   - C++11或以上 编译器

2. **编译**
    在Ubuntu 20操作系统上编译测试通过
   ```bash
   make all
   ./test_gpubigint

参考了来自https://github.com/bingshen/cuda-bigint的代码
