#ifndef LIBYACUDECU_H
#define LIBYACUDECU_H

unsigned int deconv_device(unsigned int iter, unsigned int N1, unsigned int N2, unsigned int N3, float *h_image, float *h_psf, float *h_object);
unsigned int deconv_host(unsigned int iter, unsigned int N1, unsigned int N2, unsigned int N3, float *h_image, float *h_psf, float *h_object);
unsigned int deconv_stream(unsigned int iter, unsigned int N1, unsigned int N2, unsigned int N3, float *h_image, float *h_psf, float *h_object);

#endif
