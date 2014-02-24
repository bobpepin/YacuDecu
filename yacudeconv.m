function O = yacudeconv(I, psf, varargin)
% YACUDECONV Deconvolvolve image using Richardson-Lucy method with CUDA
% J = yacudeconv(I, PSF) 
% J = yacudeconv(I, PSF, NUMIT)
% J = yacudeconv(I, PSF, NUMIT, INIT)
% deconvolves image I using Richardson-Lucy algorithm, returning 
% deconvolved image J. The assumption is that the image I was created 
% by convolving a true image with a point-spread function PSF and 
% possibly by adding noise.
%
% I and PSF must be 3-Dimensional arrays.
%
% NUMIT (optional) is the number of iterations (default is 10)
%
% INIT (optional) is the initial guess for the iterative algorithm

if nargin < 3 || isempty(varargin{1})
    numiter = 10;
else
    numiter = varargin{1};
end

if nargin < 4 || isempty(varargin{2})
    init = I;
else
    init = varargin{2};
end

if nargin < 5 || isempty(varargin{3})
    flavour = 'stream';
else
    flavour = varargin{3};
end

I = single(I);

psf = single(psf);

h = ifftshift(padarray(padarray(psf, floor((size(I) - size(psf))/2), 'circular'), mod(size(I)-size(psf), 2), 'circular', 'pre'));

if ~libisloaded('libyacudecu')
    if isdeployed
        loadlibrary('libyacudecu', @yacudecu_proto);
    else
        loadlibrary('libyacudecu', 'yacudecu.h');
    end
end

%init = imfilter(I, fspecial('gaussian'));
%init = I;
%sum(~isfinite(init(:)))
%sum(~isfinite(h(:)))

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

