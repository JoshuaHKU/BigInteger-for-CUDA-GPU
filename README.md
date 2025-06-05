# cuda-bigint

## Project Overview

`cuda-bigint` is a high-performance big integer arithmetic library based on CUDA, supporting high-precision integer addition, subtraction, multiplication, division, modulo, comparison, modular exponentiation, and bitwise operations on the GPU. The project aims to provide efficient GPU-side big integer operations as a foundation for cryptographic algorithms such as RSA and ECC.

## Main Features

- **High-precision big integer arithmetic**: Addition, subtraction, multiplication, division, modulo, comparison, bitwise operations—all supporting GPU parallel computation.
- **Modular exponentiation**: Supports big integer power-mod operations, suitable for cryptographic scenarios.
- **Multiplicative inverse**: Supports calculation of the multiplicative inverse of big integers.
- **Base conversion and string initialization**: Supports conversion between decimal, hexadecimal, and other bases, as well as initialization from strings.
- **Random big integer generation**: Supports bitwise random big integer generation, can be combined with GMP to generate large primes.
- **Unified host and device interface**: All core operations can be called on both CPU and GPU, making testing and integration convenient.

## Code Structure

- `cuda_bigint.h`  
  Core data structures and algorithm implementations for big integers, including the KernelBigInt and GPUBigInt classes and all high-precision arithmetic functions.
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
  大整数核心数据结构与算法实现，包括 KernelBigInt、GPUBigInt 类及所有高精度运算函数。
- `test_gpubigint.cu`  
  GPU 端大整数运算的功能测试，包括加减乘除、模运算、乘法逆元等典型用例，验证 CUDA 设备端的正确性和性能。


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