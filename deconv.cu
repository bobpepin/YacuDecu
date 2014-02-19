#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cufft.h>
#include <cuComplex.h>

__global__ void ComplexMul(cuComplex *A, cuComplex *B, cuComplex *C)
{
    unsigned int i = blockIdx.x * gridDim.y * gridDim.z * blockDim.x + blockIdx.y * gridDim.z * blockDim.x + blockIdx.z * blockDim.x + threadIdx.x;
    C[i] = cuCmulf(A[i], B[i]);
}

__global__ void FloatDiv(float *A, float *B, float *C)
{
    unsigned int i = blockIdx.x * gridDim.y * gridDim.z * blockDim.x + blockIdx.y * gridDim.z * blockDim.x + blockIdx.z * blockDim.x + threadIdx.x;
    C[i] = A[i] / B[i];
}

__global__ void FloatMul(float *A, float *B, float *C)
{
    unsigned int i = blockIdx.x * gridDim.y * gridDim.z * blockDim.x + blockIdx.y * gridDim.z * blockDim.x + blockIdx.z * blockDim.x + threadIdx.x;
    C[i] = A[i] * B[i];
}

extern "C" {
    int deconv_device(unsigned int iter, size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_object);
    int deconv_host(unsigned int iter, size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_object);
    int deconv_stream(unsigned int iter, size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_object);
}

static cufftResult createPlans(size_t, size_t, size_t, cufftHandle *planR2C, cufftHandle *planC2R, void **workArea, size_t *workSize);
static cudaError_t numBlocksThreads(unsigned int N, dim3 *numBlocks, dim3 *threadsPerBlock);

static float floatMean(float *a, int N) {
    float m = 0;
    for(float *p = a; p < a+N; p++) {
        m += *p;
    }
    return m / (float)N;
}

static float devFloatMean(float *a_dev, int N) {
    float *a = (float*)malloc(N*sizeof(float));
    cudaMemcpy(a, a_dev, N*sizeof(float), cudaMemcpyDeviceToHost);
    float m = floatMean(a, N);
    free(a);
    return m;
}

int deconv_device(unsigned int iter, size_t N1, size_t N2, size_t N3, 
                  float *h_image, float *h_psf, float *h_object) {
    int retval = 0;
    cufftResult r;
    cudaError_t err;
    cufftHandle planR2C, planC2R;

    float *image = 0; // convolved image (constant)
    float *object = 0; // estimated object
    cuComplex *otf = 0; // Fourier transform of PSF (constant)
    void *buf = 0; // intermediate results
    void *workArea = 0; // cuFFT work area

    size_t nSpatial = N1*N2*N3; // number of values in spatial domain
    size_t nFreq = N1*N2*(N3/2+1); // number of values in frequency domain
    //size_t nFreq = N1*(N2/2+1); // number of values in frequency domain
    size_t mSpatial, mFreq;

    dim3 freqThreadsPerBlock, spatialThreadsPerBlock, freqBlocks, spatialBlocks;
    size_t workSize; // size of cuFFT work area in bytes

    err = numBlocksThreads(nSpatial, &spatialBlocks, &spatialThreadsPerBlock);
    if(err) goto cudaErr;
    err = numBlocksThreads(nFreq, &freqBlocks, &freqThreadsPerBlock);
    if(err) goto cudaErr;

    mSpatial = spatialBlocks.x * spatialBlocks.y * spatialBlocks.z * spatialThreadsPerBlock.x * sizeof(float);
    mFreq = freqBlocks.x * freqBlocks.y * freqBlocks.z * freqThreadsPerBlock.x * sizeof(cuComplex);

    printf("N: %ld, M: %ld\n", nSpatial, mSpatial);
    printf("Blocks: %d x %d x %d, Threads: %d x %d x %d\n", spatialBlocks.x, spatialBlocks.y, spatialBlocks.z, spatialThreadsPerBlock.x, spatialThreadsPerBlock.y, spatialThreadsPerBlock.z);

    cudaDeviceReset();

    cudaProfilerStart();

    err = cudaMalloc(&image, mSpatial);
    if(err) goto cudaErr;
    err = cudaMalloc(&object, mSpatial);
    if(err) goto cudaErr;
    err = cudaMalloc(&otf, mFreq);
    if(err) goto cudaErr;
    err = cudaMalloc(&buf, mFreq); // mFreq > mSpatial
    if(err) goto cudaErr;

    err = cudaMemset(image, 0, mSpatial);
    if(err) goto cudaErr;
    err = cudaMemset(object, 0, mSpatial);
    if(err) goto cudaErr;

    printf("Memory allocated.\n");

    err = cudaMemcpy(image, h_image, nSpatial*sizeof(float), cudaMemcpyHostToDevice);
    if(err) goto cudaErr;
    printf("Image transferred.\n");
    err = cudaMemcpy(object, h_object, nSpatial*sizeof(float), cudaMemcpyHostToDevice);
    if(err) goto cudaErr;
    printf("Object transferred.\n");

    err = cudaMemcpy(otf, h_psf, nSpatial*sizeof(float), cudaMemcpyHostToDevice);
    if(err) goto cudaErr;
    printf("PSF transferred.\n");

    r = createPlans(N1, N2, N3, &planR2C, &planC2R, &workArea, &workSize);
    if(r) goto cufftError;

    printf("Plans created.\n");

    r = cufftExecR2C(planR2C, (float*)otf, otf);
    if(r) goto cufftError;

    for(unsigned int i=0; i < iter; i++) {
        printf("Iteration %d\n", i);
        r = cufftExecR2C(planR2C, object, (cufftComplex*)buf);
        if(r) goto cufftError;
        ComplexMul<<<freqBlocks, freqThreadsPerBlock>>>((cuComplex*)buf, otf, (cuComplex*)buf);
        r = cufftExecC2R(planC2R, (cufftComplex*)buf, (float*)buf);
        if(r) goto cufftError;
        FloatDiv<<<spatialBlocks, spatialThreadsPerBlock>>>(image, (float*)buf, (float*)buf);
        r = cufftExecR2C(planR2C, (float*)buf, (cufftComplex*)buf);
        if(r) goto cufftError;
        ComplexMul<<<freqBlocks, freqThreadsPerBlock>>>((cuComplex*)buf, otf, (cuComplex*)buf);
        r = cufftExecC2R(planC2R, (cufftComplex*)buf, (float*)buf);
        if(r) goto cufftError;
        FloatMul<<<spatialBlocks, spatialThreadsPerBlock>>>((float*)buf, object, object);
    }

    err = cudaMemcpy(h_object, object, nSpatial*sizeof(float), cudaMemcpyDeviceToHost);
    if(err) goto cudaErr;

    retval = 0;
    goto cleanup;

cudaErr:
    fprintf(stderr, "CUDA error: %d\n", err);
    retval = err;
    goto cleanup;

cufftError:
    fprintf(stderr, "CuFFT error: %d\n", r);
    retval = r;
    goto cleanup;

cleanup:
    if(image) cudaFree(image);
    if(object) cudaFree(object);
    if(otf) cudaFree(otf);
    if(buf) cudaFree(buf);
    if(workArea) cudaFree(workArea);
    cudaProfilerStop();
    cudaDeviceReset();
    return retval;
}

int deconv_host(unsigned int iter, size_t N1, size_t N2, size_t N3, 
                float *h_image, float *h_psf, float *h_object) {
    int retval = 0;
    cufftResult r;
    cudaError_t err;
    cufftHandle planR2C, planC2R;

    float *image = 0; // convolved image (constant)
    float *object = 0; // estimated object
    cuComplex *otf = 0; // Fourier transform of PSF (constant)
    void *buf = 0; // intermediate results
    void *workArea = 0; // cuFFT work area
    cuComplex *h_otf = 0;
    void *h_buf = 0;

    float *h_image_pad = 0;
    float *h_object_pad = 0;

    size_t nSpatial = N1*N2*N3; // number of values in spatial domain
    size_t nFreq = N1*N2*(N3/2+1); // number of values in frequency domain
    //size_t nFreq = N1*(N2/2+1); // number of values in frequency domain
    size_t mSpatial, mFreq;

    dim3 freqThreadsPerBlock, spatialThreadsPerBlock, freqBlocks, spatialBlocks;
    size_t workSize; // size of cuFFT work area in bytes

    err = numBlocksThreads(nSpatial, &spatialBlocks, &spatialThreadsPerBlock);
    if(err) goto cudaErr;
    err = numBlocksThreads(nFreq, &freqBlocks, &freqThreadsPerBlock);
    if(err) goto cudaErr;

    mSpatial = spatialBlocks.x * spatialBlocks.y * spatialBlocks.z * spatialThreadsPerBlock.x * sizeof(float);
    mFreq = freqBlocks.x * freqBlocks.y * freqBlocks.z * freqThreadsPerBlock.x * sizeof(cuComplex);

    printf("N: %ld, M: %ld\n", nSpatial, mSpatial);
    printf("Blocks: %d x %d x %d, Threads: %d x %d x %d\n", spatialBlocks.x, spatialBlocks.y, spatialBlocks.z, spatialThreadsPerBlock.x, spatialThreadsPerBlock.y, spatialThreadsPerBlock.z);

    cudaDeviceReset();
    err = cudaSetDeviceFlags(cudaDeviceMapHost);
    printf("Set Device Flags: %d\n", err);

    cudaProfilerStart();

    err = cudaHostAlloc(&h_otf, mFreq, cudaHostAllocMapped | cudaHostAllocWriteCombined);
    if(err) goto cudaErr;
    err = cudaHostAlloc(&h_buf, mFreq, cudaHostAllocMapped | cudaHostAllocWriteCombined);
    if(err) goto cudaErr;

    printf("Host memory allocated.\n");

    if(mSpatial > nSpatial*sizeof(float)) {
        err = cudaHostAlloc(&h_image_pad, mSpatial, cudaHostAllocMapped | cudaHostAllocWriteCombined);
        if(err) goto cudaErr;
        err = cudaHostAlloc(&h_object_pad, mSpatial, cudaHostAllocMapped | cudaHostAllocWriteCombined);
        if(err) goto cudaErr;
        err = cudaHostGetDevicePointer(&image, h_image_pad, 0);
        if(err) goto cudaErr;
        err = cudaHostGetDevicePointer(&object, h_object_pad, 0);
        if(err) goto cudaErr;
        err = cudaMemcpy(image, h_image, nSpatial*sizeof(float), cudaMemcpyHostToDevice);
        if(err) goto cudaErr;
        err = cudaMemcpy(object, h_object, nSpatial*sizeof(float), cudaMemcpyHostToDevice);
        if(err) goto cudaErr;
    } else {
        err = cudaHostRegister(h_image, mSpatial, cudaHostRegisterMapped);
        if(err) goto cudaErr;
        err = cudaHostRegister(h_object, mSpatial, cudaHostRegisterMapped);
        if(err) goto cudaErr;
        err = cudaHostGetDevicePointer(&image, h_image, 0);
        if(err) goto cudaErr;
        err = cudaHostGetDevicePointer(&object, h_object, 0);
        if(err) goto cudaErr;
    }
    err = cudaHostGetDevicePointer(&otf, h_otf, 0);
    if(err) goto cudaErr;
    err = cudaHostGetDevicePointer(&buf, h_buf, 0);
    if(err) goto cudaErr;

    printf("Host pointers registered.\n");

    err = cudaMemcpy(otf, h_psf, nSpatial*sizeof(float), cudaMemcpyHostToDevice);
    if(err) goto cudaErr;
    printf("PSF transferred.\n");

    r = createPlans(N1, N2, N3, &planR2C, &planC2R, &workArea, &workSize);
    if(r) goto cufftError;

    printf("Plans created.\n");

    r = cufftExecR2C(planR2C, (float*)otf, otf);
    if(r) goto cufftError;

    for(unsigned int i=0; i < iter; i++) {
        printf("Iteration %d\n", i);
        r = cufftExecR2C(planR2C, object, (cufftComplex*)buf);
        if(r) goto cufftError;
        ComplexMul<<<freqBlocks, freqThreadsPerBlock>>>((cuComplex*)buf, otf, (cuComplex*)buf);
        r = cufftExecC2R(planC2R, (cufftComplex*)buf, (float*)buf);
        if(r) goto cufftError;
        printf("a: m = %f\n", devFloatMean((float*)buf, nSpatial));
        FloatDiv<<<spatialBlocks, spatialThreadsPerBlock>>>(image, (float*)buf, (float*)buf);
        r = cufftExecR2C(planR2C, (float*)buf, (cufftComplex*)buf);
        if(r) goto cufftError;
        ComplexMul<<<freqBlocks, freqThreadsPerBlock>>>((cuComplex*)buf, otf, (cuComplex*)buf);
        r = cufftExecC2R(planC2R, (cufftComplex*)buf, (float*)buf);
        if(r) goto cufftError;
        FloatMul<<<spatialBlocks, spatialThreadsPerBlock>>>((float*)buf, object, object);
    }

    printf("object: m = %f\n", devFloatMean((float*)object, nSpatial));

    err = cudaMemcpy(h_object, object, nSpatial*sizeof(float), cudaMemcpyDeviceToHost);
    if(err) goto cudaErr;

    retval = 0;
    goto cleanup;

cudaErr:
    fprintf(stderr, "CUDA error: %d\n", err);
    retval = err;
    goto cleanup;

cufftError:
    fprintf(stderr, "CuFFT error: %d\n", r);
    retval = r;
    goto cleanup;

cleanup:
    printf("h_image: %p, h_object: %p, h_psf: %p, h_buf: %p, h_otf: %p\n", h_image, h_object, h_psf, h_buf, h_otf);
    if(image) {
        if(h_image_pad) {
            cudaHostUnregister(h_image_pad);
            cudaFreeHost(h_image_pad);
        } else {
            cudaHostUnregister(h_image);
        }
    }
    if(object) {
        if(h_object_pad) {
            cudaHostUnregister(h_object_pad);
            cudaFreeHost(h_object_pad);
        } else {
            cudaHostUnregister(h_object);
        }
    }
    if(otf) {
        cudaHostUnregister(h_otf);
        cudaFreeHost(h_otf);
    }
    if(buf) {
	cudaHostUnregister(h_buf);
	cudaFreeHost(h_buf);
    }
    if(workArea) cudaFree(workArea);
    cudaProfilerStop();
    cudaDeviceReset();
    return retval;
}

int deconv_stream(unsigned int iter, size_t N1, size_t N2, size_t N3, 
                  float *h_image, float *h_psf, float *h_object) {
    int retval = 0;
    cufftResult r;
    cudaError_t err;
    cufftHandle planR2C, planC2R;

    cudaStream_t fftStream = 0, memStream = 0;

    void *result = 0; // estimated object
    void *buf = 0; // intermediate results
    void *workArea = 0; // cuFFT work area
    cuComplex *h_otf = 0;

    size_t nSpatial = N1*N2*N3; // number of values in spatial domain
    size_t nFreq = N1*N2*(N3/2+1); // number of values in frequency domain
    //size_t nFreq = N1*(N2/2+1); // number of values in frequency domain
    size_t mSpatial, mFreq;
    size_t workSize; // size of cuFFT work area in bytes

    dim3 freqThreadsPerBlock, spatialThreadsPerBlock, freqBlocks, spatialBlocks;

    err = numBlocksThreads(nSpatial, &spatialBlocks, &spatialThreadsPerBlock);
    if(err) goto cudaErr;
    err = numBlocksThreads(nFreq, &freqBlocks, &freqThreadsPerBlock);
    if(err) goto cudaErr;

    mSpatial = spatialBlocks.x * spatialBlocks.y * spatialBlocks.z * spatialThreadsPerBlock.x * sizeof(float);
    mFreq = freqBlocks.x * freqBlocks.y * freqBlocks.z * freqThreadsPerBlock.x * sizeof(cuComplex);

    printf("N: %ld, M: %ld\n", nSpatial, mSpatial);
    printf("Blocks: %d x %d x %d, Threads: %d x %d x %d\n", spatialBlocks.x, spatialBlocks.y, spatialBlocks.z, spatialThreadsPerBlock.x, spatialThreadsPerBlock.y, spatialThreadsPerBlock.z);

    cudaDeviceReset();
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    err = cudaStreamCreate(&fftStream);
    if(err) goto cudaErr;
    err = cudaStreamCreate(&memStream);
    if(err) goto cudaErr;
#if 0
    err = cudaEventCreateWithFlags(&memSync, cudaEventDisableTiming);
    if(err) goto cudaErr;
#endif

    cudaProfilerStart();

    err = cudaMalloc(&result, mFreq);
    if(err) goto cudaErr;
    err = cudaMalloc(&buf, mFreq); // mFreq > mSpatial
    if(err) goto cudaErr;

    h_otf = (cuComplex*)malloc(nFreq*sizeof(cuComplex));

    printf("Memory allocated.\n");

    err = cudaHostRegister(h_image, nSpatial*sizeof(float), 0);
    if(err) goto cudaErr;
    err = cudaHostRegister(h_object, nSpatial*sizeof(float), 0);
    if(err) goto cudaErr;
    err = cudaHostRegister(h_otf, nFreq*sizeof(cuComplex), 0);
    if(err) goto cudaErr;

    r = createPlans(N1, N2, N3, &planR2C, &planC2R, &workArea, &workSize);
    if(r) goto cufftError;

    r = cufftSetStream(planR2C, fftStream);
    if(r) goto cufftError;
    r = cufftSetStream(planC2R, fftStream);
    if(r) goto cufftError;

    printf("Plans created.\n");

    err = cudaMemcpyAsync(buf, h_psf, nSpatial*sizeof(float), cudaMemcpyHostToDevice, fftStream);
    if(err) goto cudaErr;
    r = cufftExecR2C(planR2C, (float*)buf, (cuComplex*)buf);
    if(r) goto cufftError;
    err = cudaMemcpyAsync(h_otf, buf, nFreq*sizeof(cuComplex), cudaMemcpyDeviceToHost, fftStream);

    err = cudaStreamSynchronize(fftStream);
    if(err) goto cudaErr;

    printf("OTF generated.\n");

    err = cudaMemcpyAsync(result, h_object, nSpatial*sizeof(float), cudaMemcpyHostToDevice, fftStream);
    if(err) goto cudaErr;

    for(unsigned int i=0; i < iter; i++) {
        printf("Iteration %d\n", i);
        err = cudaMemcpyAsync(buf, h_otf, nFreq*sizeof(cuComplex), cudaMemcpyHostToDevice, memStream);
        if(err) goto cudaErr;
        r = cufftExecR2C(planR2C, (float*)result, (cuComplex*)result);
        if(r) goto cufftError;
	
        cudaDeviceSynchronize();

        ComplexMul<<<freqBlocks, freqThreadsPerBlock, 0, fftStream>>>((cuComplex*)result, (cuComplex*)buf, (cuComplex*)result);

        cudaDeviceSynchronize();

        err = cudaMemcpyAsync(buf, h_image, nSpatial*sizeof(float), cudaMemcpyHostToDevice, memStream);
        if(err) goto cudaErr;
        r = cufftExecC2R(planC2R, (cuComplex*)result, (float*)result);
        if(r) goto cufftError;

        cudaDeviceSynchronize();

        FloatDiv<<<spatialBlocks, spatialThreadsPerBlock, 0, fftStream>>>((float*)buf, (float*)result, (float*)result);

        cudaDeviceSynchronize();

        err = cudaMemcpyAsync(buf, h_otf, nFreq*sizeof(cuComplex), cudaMemcpyHostToDevice, memStream);
        if(err) goto cudaErr;
        r = cufftExecR2C(planR2C, (float*)result, (cuComplex*)result);
        if(r) goto cufftError;

        cudaDeviceSynchronize();

        ComplexMul<<<freqBlocks, freqThreadsPerBlock, 0, fftStream>>>((cuComplex*)result, (cuComplex*)buf, (cuComplex*)result);

        cudaDeviceSynchronize();

        err = cudaMemcpyAsync(buf, h_object, nSpatial*sizeof(float), cudaMemcpyHostToDevice, memStream);
        if(err) goto cudaErr;
        r = cufftExecC2R(planC2R, (cuComplex*)result, (float*)result);
        if(r) goto cufftError;

        cudaDeviceSynchronize();

        FloatMul<<<spatialBlocks, spatialThreadsPerBlock, 0, fftStream>>>((float*)buf, (float*)result, (float*)result);

        cudaDeviceSynchronize();

        err = cudaMemcpyAsync(h_object, result, nSpatial*sizeof(float), cudaMemcpyDeviceToHost, fftStream);
        if(err) goto cudaErr;
    }

    cudaDeviceSynchronize();

    retval = 0;
    goto cleanup;

cudaErr:
    fprintf(stderr, "CUDA error: %d\n", err);
    retval = err;
    goto cleanup;

cufftError:
    fprintf(stderr, "CuFFT error: %d\n", r);
    retval = r;
    goto cleanup;

cleanup:
    if(fftStream) cudaStreamDestroy(fftStream);
    if(memStream) cudaStreamDestroy(memStream);
    if(result) cudaFree(result);
    if(buf) cudaFree(buf);
    if(workArea) cudaFree(workArea);
    if(h_otf) {
        cudaHostUnregister(h_otf);
        free(h_otf);
    }
    cudaHostUnregister(h_image);
    cudaHostUnregister(h_object);
    cudaProfilerStop();
    cudaDeviceReset();
    return retval;
}

cufftResult createPlans(size_t N1, size_t N2, size_t N3, cufftHandle *planR2C, cufftHandle *planC2R, void **workArea, size_t *workSize) {
    cufftResult r;

    r = cufftCreate(planR2C);
    if(r) return r;
    r = cufftSetCompatibilityMode(*planR2C, CUFFT_COMPATIBILITY_NATIVE);
    if(r) return r;
    r = cufftSetAutoAllocation(*planR2C, 0);
    if(r) return r;

    r = cufftCreate(planC2R);
    if(r) return r;
    r = cufftSetCompatibilityMode(*planC2R, CUFFT_COMPATIBILITY_NATIVE);
    if(r) return r;
    r = cufftSetAutoAllocation(*planC2R, 0);
    if(r) return r;

    size_t tmp;
    r = cufftGetSize3d(*planR2C, N1, N2, N3, CUFFT_R2C, workSize);
    //r = cufftGetSize2d(*planR2C, N1, N2, CUFFT_R2C, workSize);
    if(r) return r;
    r = cufftGetSize3d(*planC2R, N1, N2, N3, CUFFT_R2C, &tmp);
    //r = cufftGetSize2d(*planC2R, N1, N2, CUFFT_R2C, &tmp);
    if(r) return r;
    if(tmp > *workSize)
        *workSize = tmp;

    cudaError_t err = cudaMalloc(workArea, *workSize);
    if(err) return CUFFT_ALLOC_FAILED;

    r = cufftSetWorkArea(*planR2C, *workArea);
    if(r) goto error;
    r = cufftMakePlan3d(*planR2C, N1, N2, N3, CUFFT_R2C, &tmp);
    //r = cufftMakePlan2d(*planR2C, N1, N2, CUFFT_R2C, &tmp);
    if(r) goto error;

    r = cufftSetWorkArea(*planC2R, *workArea);
    if(r) goto error;
    r = cufftMakePlan3d(*planC2R, N1, N2, N3, CUFFT_C2R, &tmp);
    //r = cufftMakePlan2d(*planC2R, N1, N2, CUFFT_C2R, &tmp);
    if(r) goto error;

    return CUFFT_SUCCESS;
error:
    cudaFree(*workArea);
    return r;
}

static cudaError_t numBlocksThreads(unsigned int N, dim3 *numBlocks, dim3 *threadsPerBlock) {
    unsigned int BLOCKSIZE = 128;
    int Nx, Ny, Nz;
    int device;
    cudaError_t err;
    if(N < BLOCKSIZE) {
        numBlocks->x = 1;
        numBlocks->y = 1;
        numBlocks->z = 1;
        threadsPerBlock->x = N;
        threadsPerBlock->y = 1;
        threadsPerBlock->z = 1;
        return cudaSuccess;
    }
    threadsPerBlock->x = BLOCKSIZE;
    threadsPerBlock->y = 1;
    threadsPerBlock->z = 1;
    err = cudaGetDevice(&device);
    if(err) return err;
    err = cudaDeviceGetAttribute(&Nx, cudaDevAttrMaxBlockDimX, device);
    if(err) return err;
    err = cudaDeviceGetAttribute(&Ny, cudaDevAttrMaxBlockDimY, device);
    if(err) return err;
    err = cudaDeviceGetAttribute(&Nz, cudaDevAttrMaxBlockDimZ, device);
    if(err) return err;
    printf("Nx: %d, Ny: %d, Nz: %d\n", Nx, Ny, Nz);
    unsigned int n = (N-1) / BLOCKSIZE + 1;
    unsigned int x = (n-1) / (Ny*Nz) + 1;
    unsigned int y = (n-1) / (x*Nz) + 1;
    unsigned int z = (n-1) / (x*y) + 1;
    if(x > Nx || y > Ny || z > Nz) {
        return cudaErrorInvalidConfiguration;
    }
    numBlocks->x = x;
    numBlocks->y = y;
    numBlocks->z = z;

    return cudaSuccess;
}
