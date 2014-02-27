YacuDecu
========

CUDA Deconvolution Library for 3D images with C and Matlab API.

This library implements the Richardson-Lucy algorithm on CUDA GPUs.

Installation
============

The software has been tested on 64bit Linux and Windows 7.

Binaries are available at https://github.com/bobpepin/YacuDecu/releases.

If you want to compile the software yourself:

- Install the CUDA SDK (>= 5.5) from nVidia
- On Linux: make -f Makefile.linux
- On Windows (with Microsoft Visual C++): nmake -f Makefile.windows

This will give you the corresponding dynamic library (.so on Linux, .dll on
Windows), which you can link into your C programs or call from within Matlab.
The library will be statically linked to the CUDA runtime but dynamically to
the CUFFT library, so make sure you link to CUFFT as well when compiling your
own programs using the library.

To use with Matlab, copy the .m, .h and .so/.dll files into a directory on the
Matlab path and make sure Matlab can find the CUFFT dll (for example, by
copying it into the same directory). The Matlab interface is described in the
corresponding section.

Usage
=====

Matlab interface
----------------

Two Matlab functions are provided: "yacudeconv" and "psf_lscm". The
"yacudeconv" function deconvolves an image using a known point-spread function.
The "psf_lscm" generates a Gaussian approximation of the point-spread function
for a laser-scanning confocal microscope.

```
  yacudeconv Deconvolvolve image using Richardson-Lucy method with CUDA
  J = yacudeconv(I, PSF) 
  J = yacudeconv(I, PSF, NUMIT)
  deconvolves image I using Richardson-Lucy algorithm, returning 
  deconvolved image J. The assumption is that the image I was created 
  by convolving a true image with a point-spread function PSF and 
  possibly by adding noise. The central pixel in the PSF array corresponds
  to the center of the psf.
  I and PSF must be 3-Dimensional arrays. 
  NUMIT (optional) is the number of iterations (default is 10)

  psf_lscm Compute a Gaussian PSF for a Laser-Scanning Confocal Microscope
  P = psf_lscm(LATERAL, AXIAL, LAMBDA_EX, LAMBDA_EM, NA, N, D)
  LATERAL and AXIAL are the lateral and axial spacings (in metres).
  LAMBDA_EX and LAMBDA_EM are the excitation and emission wavelengths (in
  metres).
  NA is the microscope objective Numerical Aperture.
  N is the sample medium refractive index.
  D is the pinhole diameter in Airy Units.
```

C interface
-----------

The C interface exposes three different implementations of the algorithm,
"deconv_device", "deconv_stream" and "deconv_host", corresponding to different
trade-offs between speed and GPU memory usage. The "deconv_device"
implementation keeps all the data in GPU memory. The "deconv_stream"
implementation keeps intermediate results of the Richardson-Lucy algorithm in
GPU memory and copies the PSF, the convolved image and the result from the
previous iteration into GPU memory as needed, using streams to overlap memory
transfers with FFT computations. The "deconv_device" implementation finally
uses the capacity of newer GPUs to directly access host memory to only keep the
FFT buffer in GPU memory.

All three implementations have the same interface:

```
unsigned int deconv_{device,stream,host}(unsigned int iter, 
                                         unsigned int N1, 
                                         unsigned int N2, 
                                         unsigned int N3, 
                                         float *h_image, 
                                         float *h_psf, 
                                         float *h_object);
```
 

* iter is the number of iterations
* N1, N2 and N3 are the dimensions of the image, psf and object buffers
* h_image, h_psf and h_object are pointers to memory regions containing the
  corresponding data. The psf is assumed to be in transfer-function format: the
  pixel (0, 0, 0) of h_psf corresponds to the center point of the psf.
* The return value corresponds to either a CUDA or CUFFT error code, the
  meaning of which can be found in the nVidia documentation for the CUDA or
  CUFFT libraries respectively (cudaError_t or cufftResult types respectively).


* deconv_device uses 5 buffers of N1 x N2 x N3 32-bit floats of GPU memory.
* deconv_stream uses 3 buffers of N1 x N2 x N3 32-bit floats of GPU memory.
* deconv_host uses 1 buffer of N1 x N2 x N3 32-bit floats of GPU memory.

In my experience, the memory transfers in deconv_stream overlap well with the
FFT and the performance difference to deconv_device was small, which is why it
is the default method used in the Matlab interface. However, the Matlab
interface has a fourth parameter, which can be one of 'device', 'stream' or
'host' to use the corresponding implementation.
