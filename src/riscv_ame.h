/*
 * RISCV AME (Accelerated Matrix Extension) Runtime Header
 * Provides C intrinsics for AME matrix operations following the AME specification
 */

#ifndef RISCV_AME_H
#define RISCV_AME_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Matrix tile type - opaque handle for tile registers m0-m7
typedef void* matrix_tile_t;

// Set matrix element type
// 0 = e8 (int8), 1 = e16 (fp16), 2 = e32 (fp32)
static inline void msettype(int etype) {
    asm volatile("msettype %0" : : "r"(etype));
}

// Set tile dimensions (returns actual tile size that can be configured)
static inline size_t msettilem(size_t m) {
    size_t result;
    asm volatile("msettilem %0, %1" : "=r"(result) : "r"(m));
    return result;
}

static inline size_t msettilen(size_t n) {
    size_t result;
    asm volatile("msettilen %0, %1" : "=r"(result) : "r"(n));
    return result;
}

static inline size_t msettilek(size_t k) {
    size_t result;
    asm volatile("msettilek %0, %1" : "=r"(result) : "r"(k));
    return result;
}

// Load matrix A (M x K) with stride in bytes
static inline matrix_tile_t mlae8_m(const void* addr, size_t stride) {
    matrix_tile_t result;
    asm volatile("mlae.8.m %0, (%1), %2" : "=vm"(result) : "r"(addr), "r"(stride) : "memory");
    return result;
}

static inline matrix_tile_t mlae16_m(const void* addr, size_t stride) {
    matrix_tile_t result;
    asm volatile("mlae.16.m %0, (%1), %2" : "=vm"(result) : "r"(addr), "r"(stride) : "memory");
    return result;
}

static inline matrix_tile_t mlae32_m(const void* addr, size_t stride) {
    matrix_tile_t result;
    asm volatile("mlae.32.m %0, (%1), %2" : "=vm"(result) : "r"(addr), "r"(stride) : "memory");
    return result;
}

static inline matrix_tile_t mlae64_m(const void* addr, size_t stride) {
    matrix_tile_t result;
    asm volatile("mlae.64.m %0, (%1), %2" : "=vm"(result) : "r"(addr), "r"(stride) : "memory");
    return result;
}

// Load matrix B (K x N) with stride in bytes
static inline matrix_tile_t mlbe8_m(const void* addr, size_t stride) {
    matrix_tile_t result;
    asm volatile("mlbe.8.m %0, (%1), %2" : "=vm"(result) : "r"(addr), "r"(stride) : "memory");
    return result;
}

static inline matrix_tile_t mlbe16_m(const void* addr, size_t stride) {
    matrix_tile_t result;
    asm volatile("mlbe.16.m %0, (%1), %2" : "=vm"(result) : "r"(addr), "r"(stride) : "memory");
    return result;
}

static inline matrix_tile_t mlbe32_m(const void* addr, size_t stride) {
    matrix_tile_t result;
    asm volatile("mlbe.32.m %0, (%1), %2" : "=vm"(result) : "r"(addr), "r"(stride) : "memory");
    return result;
}

static inline matrix_tile_t mlbe64_m(const void* addr, size_t stride) {
    matrix_tile_t result;
    asm volatile("mlbe.64.m %0, (%1), %2" : "=vm"(result) : "r"(addr), "r"(stride) : "memory");
    return result;
}

// Matrix FMA: out = out + A * B (widening to double precision)
static inline matrix_tile_t mfwma_mm(matrix_tile_t a, matrix_tile_t b) {
    matrix_tile_t result;
    asm volatile("mfwma.mm %0, %1, %2" : "=vm"(result) : "vm"(a), "vm"(b));
    return result;
}

// Matrix subtraction (clear): out = a - b
static inline matrix_tile_t mwsub_mm(matrix_tile_t a, matrix_tile_t b) {
    matrix_tile_t result;
    asm volatile("mwsub.mm %0, %1, %2" : "=vm"(result) : "vm"(a), "vm"(b));
    return result;
}

// Narrowing convert from wide (double precision) to single
static inline matrix_tile_t mfncvt_f_fw_m(matrix_tile_t wide) {
    matrix_tile_t result;
    asm volatile("mfncvt.f.fw.m %0, %1" : "=vm"(result) : "vm"(wide));
    return result;
}

// Store matrix C with stride in bytes
static inline void msce8_m(matrix_tile_t src, void* addr, size_t stride) {
    asm volatile("msce.8.m %0, (%1), %2" : : "vm"(src), "r"(addr), "r"(stride) : "memory");
}

static inline void msce16_m(matrix_tile_t src, void* addr, size_t stride) {
    asm volatile("msce.16.m %0, (%1), %2" : : "vm"(src), "r"(addr), "r"(stride) : "memory");
}

static inline void msce32_m(matrix_tile_t src, void* addr, size_t stride) {
    asm volatile("msce.32.m %0, (%1), %2" : : "vm"(src), "r"(addr), "r"(stride) : "memory");
}

static inline void msce64_m(matrix_tile_t src, void* addr, size_t stride) {
    asm volatile("msce.64.m %0, (%1), %2" : : "vm"(src), "r"(addr), "r"(stride) : "memory");
}

#ifdef __cplusplus
}
#endif

#endif // RISCV_AME_H
