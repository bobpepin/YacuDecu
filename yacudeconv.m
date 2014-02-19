function O = yacudeconv(I, psf, varargin)
% YACUDECONV Deconvolvolve image using Richardson-Lucy method with CUDA
% J = yacudeconv(I, PSF) 
% J = yacudeconv(I, PSF, NUMIT)
% deconvolves image I using Richardson-Lucy algorithm, returning 
% deconvolved image J. The assumption is that the image I was created 
% by convolving a true image with a point-spread function PSF and 
% possibly by adding noise. The central pixel in the PSF array corresponds
% to the center of the psf.
%
% I and PSF must be 3-Dimensional arrays.
%
% NUMIT (optional) is the number of iterations (default is 10)

if nargin < 3
    numiter = 10;
else
    numiter = varargin{1};
end

if nargin < 4
    flavour = 'stream';
else
    flavour = varargin{2};
end

I = single(I);

psf = single(psf);

h = ifftshift(padarray(padarray(psf, floor((size(I) - size(psf))/2)), mod(size(I)-size(psf), 2), 'pre'));

if ~libisloaded('libyacudecu')
    loadlibrary('libyacudecu', 'yacudecu.h');
end

%init = imfilter(I, fspecial('gaussian'));
init = I;
[status, ~, ~, out] = calllib('libyacudecu', ['deconv_' flavour], numiter, size(I, 3), size(I, 2), size(I, 1), I, h, init);
if status == 38
    error('YacuDecu:cudaError', 'CUDA error: No CUDA devices detected (error code 38)');
elseif status == 46
    error('YacuDecu:cudaError', 'CUDA error: No CUDA devices available (error code 46)');
elseif status == 2
    error('YacuDecu:cudaError', 'CUDA error: Out of GPU memory (error code 2)');
elseif status ~= 0
    error('YacuDecu:cudaError', 'CUDA error: %d', status);
end
O = reshape(out, size(I));

end

