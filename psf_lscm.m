function [P, params] = psf_lscm(r_lateral, r_axial, lambda_ex, lambda_em, NA, n, D)
% PSF_LSCM Compute a Gaussian PSF for a Laser-Scanning Confocal Microscope
%
% P = psf_lscm(LATERAL, AXIAL, LAMBDA_EX, LAMBDA_EM, NA, N, D)
%
% LATERAL and AXIAL are the lateral and axial spacings (in metres).
%
% LAMBDA_EX and LAMBDA_EM are the excitation and emission wavelengths (in
% metres).
%
% NA is the microscope objective Numerical Aperture.
%
% N is the sample medium refractive index.
%
% D is the pinhole diameter in Airy Units.

AU = 1.22 * lambda_ex / NA;
r = D * AU / 2;

params = psf_params(lambda_ex, lambda_em, NA, n, r);

Nlat = ceil(4 * params.sigma_rho_lscm / r_lateral);
Nax = ceil(4 * params.sigma_z_lscm / r_axial);

latgrid = (-Nlat:Nlat)*r_lateral;
axgrid = (-Nax:Nax)*r_axial;

[x, y, z] = meshgrid(latgrid, latgrid, axgrid);
[theta, rho, z] = cart2pol(x, y, z);

P = exp(-0.5 .* ((rho./params.sigma_rho_lscm).^2 + (z./params.sigma_z_lscm).^2));

end

function params = psf_params(lambda_ex, lambda_em, NA, n, r)

k_ex = 2*pi / lambda_ex;
k_em = 2*pi / lambda_em;

cosa = cos(asin(NA/n));

sr = ((4 - 7*cosa^(3/2) + 3*cosa^(7/2))/(7*(1-cosa^(3/2))))^(-1/2);
s_em_r = sr / (n*k_em);
s_ex_r = sr / (n*k_ex);

e = exp((r/s_em_r)^2/2)-1;
sigma_r = sqrt(2) * ( (2*s_em_r^4*e + r^2*s_ex_r^2) / ...
                  (s_ex_r^2*s_em_r^4*e) )^(-1/2);

params.sigma_rho_wffm = s_em_r;
params.sigma_rho_lscm = sigma_r;

sz = (5 * sqrt(7) * (1-cosa^(3/2))) / ...
     (sqrt(6)* n * (4*cosa^5 - 25*cosa^(7/2) + 42*cosa^(5/2) - 25*cosa^(3/2) + 4)^(1/2));
s_em_z = sz / k_em;
s_ex_z = sz / k_ex;

sigma_z = (s_ex_z * s_em_z) / (s_ex_z^2 + s_em_z^2)^(1/2);

params.sigma_z_wffm = s_em_z;
params.sigma_z_lscm = sigma_z;

end
