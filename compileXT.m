% This generates yacudecu_proto.m and libyacudecu_thunk_pcwin64.dll
loadlibrary('libyacudecu','yacudecu.h', 'mfilename', 'yacudecu_proto.m')

% This uses the default paths to the CUFFT DLL and the Imaris JAR, adjust to your system
mcc -v -m XTYacuDeconv.m -a libyacudecu_thunk_pcwin64.dll -a 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\bin\cufft64_55.dll' -a libyacudecu.dll -a 'C:\Program Files\Bitplane\Imaris x64 7.6.5\XT\rtmatlab\ImarisLib.jar'