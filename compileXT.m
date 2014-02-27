% Pre-Requisites, make sure these are on the Matlab path before starting compilation:
% IceImarisConnector (>= 0.3.3): http://www.scs2.net/next/index.php?id=110
% GUI Layout Toolbox: http://www.mathworks.com/matlabcentral/fileexchange/27758-gui-layout-toolbox

% This generates yacudecu_proto.m and libyacudecu_thunk_pcwin64.dll
loadlibrary('libyacudecu','yacudecu.h', 'mfilename', 'yacudecu_proto.m')

copyfile('C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\bin\cufft64_55.dll', '.');

% This uses the default paths to the CUFFT DLL and the Imaris JAR, adjust to match your system
mcc -v -m XTYacuDeconv.m -a libyacudecu_thunk_pcwin64.dll -a cufft64_55.dll -a libyacudecu.dll -a 'C:\Program Files\Bitplane\Imaris x64 7.6.5\XT\rtmatlab\ImarisLib.jar'