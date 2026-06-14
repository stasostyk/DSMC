#ifndef CUDA_TEST_H
#define CUDA_TEST_H

#ifdef __cplusplus
extern "C" {
#endif

void vectorAdd(const float* a, const float* b, float* c, int n);

#ifdef __cplusplus
}
#endif

#endif // CUDA_TEST_H