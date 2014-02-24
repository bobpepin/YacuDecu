%
%
%  CUDA Deconvolution
%
%  Copyright Bob Pepin 2014
%
%
%  Installation:
%
%  - Copy this file into the XTensions folder in the Imaris installation directory
%  - You will find this function in the Image Processing menu
%
%    <CustomTools>
%      <Menu>
%        <Item name="CUDA Deconvolution" icon="Matlab" tooltip="Deconvolve using CUDA.">
%          <Command>MatlabXT::XTYacuDeconv(%i)</Command>
%        </Item>
%      </Menu>
%    </CustomTools>
%
%
%  Description:
%
%   Deconvolve a 3D dataset.
%
%

function XTYacuDeconv(aImarisApplicationID)

if isdeployed
    thisdir = ctfroot;
else
    thisdir = fileparts( mfilename( 'fullpath' ) );
end
%addpath(fullfile(thisdir, 'YacuDecu'));
%addpath(fullfile(thisdir, 'YacuDecu/GUILayout-v1p14'));
%addpath(fullfile(thisdir, 'YacuDecu/GUILayout-v1p14/Patch'));

addpath(fullfile(thisdir, 'XTYacuDeconv'));

%fprintf('CTF Directory: %s', fullfile(thisdir, 'XTYacuDeconv'));

if isa(aImarisApplicationID, 'char')
    aImarisApplicationID = str2double(aImarisApplicationID);
end

%fprintf('Imaris Application ID: %d\n', aImarisApplicationID);

conn = IceImarisConnector(aImarisApplicationID);

[sizeX, sizeY, sizeZ, sizeC, sizeT] = conn.getSizes();

if any([sizeX sizeY sizeZ] < 2)
    errordlg('3D image required for deconvolution', ...
             'CUDA Deconvolution Error', 'modal');
    return;
end

[lateral, ~, axial] = conn.getVoxelSizes();

ds = conn.mImarisApplication.GetDataSet;
NA = str2double(ds.GetParameter('Image', 'NumericalAperture'));
N = str2double(ds.GetParameter('ZeissAttrs', 'imagedocument/metadata/information/image/objectivesettings/refractiveindex'));
for i=1:sizeC
    ch = i-1;
    color = ds.GetParameter(sprintf('Channel %d', ch), 'Color');
    channels(i).Color = str2double(cell(color.split(' ')));
    channels(i).DyeName = char(ds.GetParameter(sprintf('Channel %d', ch), 'Dye name'));
    if length(channels(i).DyeName) == 0
        channels(i).DyeName = char(ds.GetChannelName(i-1));
    end
    channels(i).lambda_em = str2double(ds.GetParameter(sprintf('Channel %d', ch), 'LSMEmissionWavelength'));
    if isnan(channels(i).lambda_em)
        em = 'imagedocument/metadata/information/image/dimensions/channels/channel/emissionwavelength %d';
        channels(i).lambda_em = str2double(ds.GetParameter('ZeissAttrs', sprintf(em, ch)));
    end
    channels(i).lambda_ex = str2double(ds.GetParameter(sprintf('Channel %d', ch), 'LSMExcitationWavelength'));
    if isnan(channels(i).lambda_ex)
        ex = 'imagedocument/metadata/information/image/dimensions/channels/channel/lightsourcesettings/lightsourcesettings/wavelength %d';
        channels(i).lambda_ex = str2double(ds.GetParameter('ZeissAttrs', sprintf(ex, ch)));
    end
    if isnan(channels(i).lambda_ex)
        ex = 'imagedocument/metadata/information/image/dimensions/channels/channel/excitationwavelength %d';
        channels(i).lambda_ex = str2double(ds.GetParameter('ZeissAttrs', sprintf(ex, ch)));
    end
    channels(i).D = str2double(ds.GetParameter(sprintf('Channel %d', ch), 'LSMPinhole'));
    if isnan(channels(i).D)
        airy = 'imagedocument/metadata/information/image/dimensions/channels/channel/pinholesizeairy %d';
        channels(i).D = str2double(ds.GetParameter('ZeissAttrs', sprintf(airy, ch)));
    end
end

image = struct('lateral', lateral, 'axial', axial, 'NA', NA, 'n', N);
[status, idata, cdata] = yacugui(image, channels);

if ~status
    return
end

r_lateral = idata.lateral * 1e-6;
r_axial = idata.axial * 1e-6;
NA = idata.NA;
n = idata.n;

h = waitbar(0, 'Starting Deconvolution');

conn.mImarisApplication.DataSetPushUndo('CUDA Deconvolution');

tic
for i=1:length(cdata)
try
    if cdata(i).deconvolve == 1
        continue
    end
    waitbar((i-1)/length(cdata), h, sprintf('Deconvolving channel %d...', i));
    channel = i-1;

    lambda_ex = cdata(i).lambda_ex * 1e-9;
    lambda_em = cdata(i).lambda_em * 1e-9;
    D = cdata(i).D;
    
    iterations = 0;
    initblur = 0;
    if cdata(i).deconvolve == 2
        psf = psf_lscm(r_lateral, r_axial, lambda_ex, lambda_em, NA, n, D);
        if ~all(isfinite(psf))
            errordlg(sprintf('PSF generation failed for channel %d, check PSF parameter values and units.', i), 'Deconvolution Error');
            %fprintf('PSF generation failed for channel %d, check PSF parameter values and units.\n', i);
            continue
        end
        iterations = cdata(i).iterations;
        initblur = cdata(i).initblur;
    elseif cdata(i).deconvolve == 3
        psf = cdata(i).psfdata(:, :, :, cdata(i).psfchan);
        iterations = cdata(i).psfiterations;
        initblur = cdata(i).psfinitblur;
%        sizes = size(psf);
%        fprintf('Loaded PSF (%dx%dx%d)', sizes(2), sizes(1), sizes(3));
    end

    t = conn.mImarisApplication.GetVisibleIndexT;
    I = conn.getDataVolumeRM(channel, t);
    sizeI = size(I);
    I = yacupad(I, size(psf));
    if initblur > 0
        init = imfilter(I, fspecial('gaussian', [ceil(6*initblur)+1, ceil(6*initblur)+1], initblur));
    else
        init = I;
    end
    O = yacudeconv(I, psf, iterations, init, 'stream');
    
%    size(I)
%    size(O)
%    sum(~isfinite(O(:)))
%    sum(isnan(O(:)))
    
    pad = size(O) - sizeI;
    pre = ceil(pad / 2);
    O = O((1+pre(1)):(pre(1)+sizeI(1)), (1+pre(2)):(pre(2)+sizeI(2)), (1+pre(3)):(pre(3)+sizeI(3)));
    

    if ~all(isfinite(O(:)))
        errordlg(sprintf('Deconvolution diverged, skipping channel %d', i), 'Deconvolution Error');
        continue
    end
    
    conn.setDataVolumeRM(cast(O, conn.getMatlabDatatype()), channel, t);
catch err
    eh = errordlg(sprintf('%s\nDeconvolution stopped.', err.getReport('basic')), 'Deconvolution Error', 'modal');
    uiwait(eh);
    rethrow(err);
end
end
toc

waitbar(1, h, 'Deconvolution finished');
close(h);

%input('Press Enter to continue.', 's');

end

function [status, idata, cdata] = yacugui(image, channels)

edprops = {'Style', 'edit', 'BackgroundColor', 'w', 'HorizontalAlign', 'left'};

iinputs = {{'lateral', 'Lateral (X-Y) Spacing (um)', {edprops{:}, 'String', sprintf('%.3f', image.lateral)}}, ...
           {'axial', 'Axial (Z) Spacing (um)', {edprops{:}, 'String', sprintf('%.3f', image.axial)}}, ...
           {'NA', 'Objective Numerical Aperture', {edprops{:}, 'String', sprintf('%.3f', image.NA)}}, ...
           {'n', 'Immersion Medium Refraction Index', {edprops{:}, 'String', sprintf('%.3f', image.n)}}};       

chinputs = {{'iterations', 'Iterations', {edprops{:}, 'String', '20'}}, ...
            {'initblur', 'Initial Blur Radius (pixels)', {edprops{:}, 'String', '0'}}, ...
            {'lambda_ex', 'Excitation Wavelength (nm)', edprops}, ...
            {'lambda_em', 'Emission Wavelength (nm)', edprops}, ...
            {'D', 'Pinhole Diameter (Airy Units)', edprops}};

ss = get(0, 'ScreenSize');
w = 350 * length(channels);
h = 600;
dlg = dialog('Resize', 'on', 'Position', [ceil((ss(3)-w)/2), ceil((ss(4)-h)/2), w, h]);
set(dlg, 'DefaultUicontrolFontSize', 12);

bigpanel = uiextras.VBox('Parent', dlg, 'Spacing', 0, 'Padding', 10);

uicontrol('Parent', bigpanel, 'Style', 'text', 'String', 'YacuDecu CUDA Deconvolution', 'FontWeight', 'bold', 'FontSize', 14);

imagepanel = uiextras.HBox('Parent', bigpanel);
uiextras.Empty('Parent', imagepanel);
ig = uiextras.Grid('Parent', imagepanel, 'Spacing', 5);
uiextras.Empty('Parent', imagepanel);
for j=1:length(iinputs)
    uicontrol('Parent', ig, 'Style', 'text', 'String', iinputs{j}{2}, 'HorizontalAlign', 'left');
end
for j=1:length(iinputs)
    [id, ~, props] = iinputs{j}{:};
    icontrols.(id) = uicontrol('Parent', ig, props{:});
end
set(ig, 'ColumnSizes', [-1 50], 'RowSizes', 25*ones(1, length(iinputs)));

set(imagepanel, 'Sizes', [-1 320 -1]);

channelpanel = uiextras.HBox('Parent', bigpanel, 'Spacing', 20, 'Padding', 20);
controls = struct([]);
for i=1:length(channels)
    vbox = uiextras.VBox('Parent', channelpanel, 'Spacing', 15);
    uicontrol('Parent', vbox, 'Style', 'text', 'String', sprintf('Channel %d: %s', i, channels(i).DyeName), 'FontWeight', 'bold');
    menu = uicontrol('Parent', vbox, 'Style', 'popupmenu', ...
        'String', {'No deconvolution', 'Computed PSF', 'Measured PSF'}, ...
        'Value', 2, ...
        'BackgroundColor', 'w');
    controls(i).menu = menu;
    card = uiextras.CardPanel('Parent', vbox);
    set(menu, 'Callback', @(ho, ed) set(card, 'SelectedChild', get(menu, 'Value')))
    
    uiextras.Empty('Parent', card);
    
    g = uiextras.Grid('Parent', card, 'Spacing', 5);
    for j=1:length(chinputs)
        uicontrol('Parent', g, 'Style', 'text', 'String', chinputs{j}{2}, 'HorizontalAlign', 'left');
    end
    for j=1:length(chinputs)
        [id, ~, props] = chinputs{j}{:};
        controls(i).(id) = uicontrol('Parent', g);
        if isfield(channels(i), id)
            set(controls(i).(id), 'String', sprintf('%.2f', channels(i).(id)));
        end
        set(controls(i).(id), props{:});
    end
    set(g, 'ColumnSizes', [-1 60], 'RowSizes', 25*ones(1, length(chinputs)));
    
    filebox = uiextras.VBox('Parent', card, 'Spacing', 5);
    itergrid = uiextras.Grid('Parent', filebox, 'Spacing', 5);
    uicontrol('Parent', itergrid, 'Style', 'text', 'String', 'Iterations', 'HorizontalAlign', 'left');
    uicontrol('Parent', itergrid, 'Style', 'text', 'String', 'Initial Blur Radius (pixels)', 'HorizontalAlign', 'left');
    controls(i).psfiterations = uicontrol('Parent', itergrid, edprops{:}, 'String', '20');
    controls(i).psfinitblur = uicontrol('Parent', itergrid, edprops{:}, 'String', '20');
    set(itergrid, 'ColumnSizes', [-1 60], 'RowSizes', 25*ones(1, 2));
%    uicontrol('Parent', filebox, 'Style', 'text', 'String', 'PSF File');
    but = uicontrol('Parent', filebox, 'Style', 'pushbutton', 'String', 'Load PSF file');
    fname = uicontrol('Parent', filebox, 'Style', 'edit', 'Enable', 'inactive');
    psfchan = uicontrol('Parent', filebox, 'Style', 'popupmenu', ...
        'String', {''}, ...
        'Value', 1, ...
        'BackgroundColor', 'w');
    controls(i).psfchan = psfchan;
    set(but, 'Callback', @(ho, ed) yacugui_loadPSF(i, fname, psfchan));

    set(filebox, 'Sizes', [2*25+5 25 25 25]);
    set(card, 'SelectedChild', 2);
    
    set(vbox, 'Sizes', [25 30 -1]);
end

bbox = uiextras.HButtonBox('Parent', bigpanel, 'ButtonSize', [150 30]);
uicontrol('Parent', bbox, 'String', 'Start Deconvolution', 'Callback', @(ho, ed) set(dlg, 'UserData', true));
uiextras.Empty('Parent', bbox);
uicontrol('Parent', bbox, 'String', 'Cancel', 'Callback', @(ho, ed) set(dlg, 'UserData', false));

uicontrol('Parent', bigpanel, 'Style', 'text', 'String', 'Copyright (c) 2014 Bob Pepin (https://github.com/bobpepin/YacuDecu)', 'FontSize', 8, 'HorizontalAlign', 'right');

set(bigpanel, 'Sizes', [-1 -4 -6 -1 15]);

guidata(dlg, struct('auxconn', {[]}, 'psfdata', {cell(1, length(channels))}));

waitfor(dlg, 'UserData');

if ~ishghandle(dlg)
    idata = [];
    cdata = [];
    status = false;
    return
end

status = get(dlg, 'UserData');

for j=1:length(iinputs)
    id = iinputs{j}{1};
    val = get(icontrols.(id), 'String');
    idata.(id) = str2double(val);
end

data = guidata(dlg);

for i=1:length(channels)
    for j=1:length(chinputs)
        id = chinputs{j}{1};
        val = get(controls(i).(id), 'String');
        cdata(i).(id) = str2double(val);
    end
    cdata(i).deconvolve = get(controls(i).menu, 'Value');
    cdata(i).psfchan = get(controls(i).psfchan, 'Value');
    cdata(i).psfdata = data.psfdata{i};
    cdata(i).psfiterations = str2double(get(controls(i).psfiterations, 'String'));
    cdata(i).psfinitblur = str2double(get(controls(i).psfinitblur, 'String'));
end

close(dlg);

end

function yacugui_loadPSF(chanidx, fname, psfchan)
data = guidata(gcbo);
%disp(data)
conn = IceImarisConnector;
conn.startImaris;
app = conn.mImarisApplication;
app.FileOpen('', '');
[sizeX, sizeY, sizeZ, sizeC] = conn.getSizes();
if sizeX > 0
%msgbox(sprintf('PSF Loaded (%d channels).', sizeC));
    set(fname, 'String', char(conn.mImarisApplication.GetCurrentFileName()));
    set(psfchan, 'String', arrayfun(@(c) sprintf('Channel %d', c), 1:sizeC, 'Uni', 0));
    I = zeros([sizeY, sizeX, sizeZ, sizeC], conn.getMatlabDatatype());
    for i=1:sizeC
        I(:, :, :, i) = conn.getDataVolumeRM(i-1, 0);
    end
    data.psfdata{chanidx} = I;
    guidata(gcbo, data);
end
figure(gcbf);
conn.closeImaris(1);
end

function yacugui_loadPSF_existing(chanidx, fname, psfchan, conn)
data = guidata(gcbo);
disp(data)
app = conn.mImarisApplication;
origfname = app.GetCurrentFileName();
ds = app.GetDataSet.Clone;
app.FileOpen('', 'LoadDataSet="eDataSetNo"');
[sizeX, sizeY, sizeZ, sizeC] = conn.getSizes();
if ~strcmp(app.GetCurrentFileName, origfname)
%msgbox(sprintf('PSF Loaded (%d channels).', sizeC));
    set(fname, 'String', char(conn.mImarisApplication.GetCurrentFileName()));
    set(psfchan, 'String', arrayfun(@(c) sprintf('Channel %d', c), 1:sizeC, 'Uni', 0));
    I = zeros([sizeY, sizeX, sizeZ, sizeC], conn.getMatlabDatatype());
    for i=1:sizeC
        I(:, :, :, i) = conn.getDataVolumeRM(i-1, 0);
    end
    data.psfdata{chanidx} = I;
    guidata(gcbo, data);
end
figure(gcbf);
app.FileOpen(origfname, 'LoadDataSet="eDataSetYes"');
app.SetDataSet(ds);
end


function yacugui_loadPSF_old(chanidx, fname, psfchan)
data = guidata(gcbo);
disp(data)
if isempty(data.auxconn)
    data.auxconn = IceImarisConnector;
    data.auxconn.startImaris;
    guidata(gcbo, data);
end
conn = data.auxconn;
conn.mImarisApplication.FileOpen('', '');
conn.mImarisApplication.SetVisible(true);
%conn.mImarisApplication.SetVisible(false);
if isempty(conn.mImarisApplication.GetDataSet)
    return
end
[sizeX, sizeY, sizeZ, sizeC] = conn.getSizes();
%msgbox(sprintf('PSF Loaded (%d channels).', sizeC));
set(fname, 'String', char(conn.mImarisApplication.GetCurrentFileName()));
set(psfchan, 'String', arrayfun(@(c) sprintf('Channel %d', c), 1:sizeC, 'Uni', 0));
I = zeros([sizeY, sizeX, sizeZ, sizeC], conn.getMatlabDatatype());
for i=1:sizeC
    I(:, :, :, i) = conn.getDataVolumeRM(i-1, 0);
end
data.psfdata{chanidx} = I;
guidata(gcbo, data);
end
