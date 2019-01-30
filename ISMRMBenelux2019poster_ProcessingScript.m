% SCRIPT: ISMRMBenelux2019poster_ProcessingScript
%--------------------------------------------------------------------------
% Copyright (C) Neu3CA Research Group, Eindhoven University of Technology
%
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; either version 3
% of the License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307,USA.
%
% Author: Stephan Heunis, <j.s.heunis@tue.nl>, 2019

%--------------------------------------------------------------------------
% DEFINITION
%--------------------------------------------------------------------------

% The "ISMRMBenelux2019poster_ProcessingScript.m" script allows real-time
% multi-echo fMRI analysis. It runs through a set of preprocessing and
% real-time processing scripts executed on resting state fMRI data of 31
% subjects from an OpenNeuro dataset. It ends by calling a separate script,
% which in turn generates the results shown in the poster presented at the
% 11th Annual meeting of the ISMRM Benelux chapter.
% See: https://doi.org/10.5281/zenodo.2553256

% The script requires the following setup/installs:
%   -   Matlab 2014a or later
%   -   SPM12: http://www.fil.ion.ucl.ac.uk/spm/
%   -   Experimental data: https://openneuro.org/datasets/ds000210
%   -   File/directory locations (see below in initialisation section)

%%

%--------------------------------------------------------------------------
% INITIALISATION (USER INPUT REQUIRED)
%--------------------------------------------------------------------------
% STEP 1 - Download data
% We use open data from OpenNeuro.org:
% https://openneuro.org/datasets/ds000210 

% STEP 2 - Set required directory variables:
% Specify Matlab directory
matlab_dir          =   ''; % E.g. '/Users/jheunis/Documents/MATLAB';
% Specify SPM installation directory
spm_dir             =   ''; % E.g. '/Users/jheunis/Documents/MATLAB/spm12';
% Specify parent directory that contains all downloaded unprocessed data
raw_data_dir        =   ''; % E.g. '/Users/jheunis/Documents/ds000210-00002;
% Specify output directory for script results
data_dir            =   ''; % E.g. '/Users/jheunis/Documents/rtme_output;

% STEP 3 - Initialise imaging, experimental and algorithmic
% variables/parameters
Nsub            = 31; % number of subjects
TR              = 3; % repetition time
Nt              = 204; % number of volumes in the functional time series
Nskip           = 4; % number of volumes to ignore, i.e. dummy volumes
Nstart          = Nskip + 1;
Ne              = 3; % number of echoes per volume
Eref            = 2; % reference echo, for realignment
TE              = [13.7 30 47]; % Echo times in ms
T2star_thresh   = 100; % threshold for maximum T2star after estimation


%%

%--------------------------------------------------------------------------
% BIDS DATA SETUP
%--------------------------------------------------------------------------

% This section copies the raw data from the 'raw_data_dir' into the
% 'data_dir', which will eventually contain all raw and processed data, as
% well as the results. First, the BIDS data directory structure is created
% and then data are copied and unzipped. This leaves raw data untouched.

if ~exist(data_dir, 'dir')
    mkdir(data_dir)
end

for subj = 1:Nsub
    subj_dir = [data_dir filesep 'sub-' sprintf('%02d',subj)]; % subject directory name
    func_dir = [subj_dir filesep 'func']; % functional directory for sub
    anat_dir = [subj_dir filesep 'anat']; % anatomical directory for sub
    if ~exist(func_dir, 'dir')
        mkdir(func_dir)
    end
    if ~exist(anat_dir, 'dir')
        mkdir(anat_dir)
    end
    
    % copy and unzip anatomical image
    raw_anat_fn = [raw_data_dir filesep 'sub-' sprintf('%02d',subj) '_anat_sub-' sprintf('%02d',subj) '_T1w.nii.gz'];
    anat_fn = [anat_dir filesep 'sub-' sprintf('%02d',subj) '_T1w.nii.gz'];
    copyfile(raw_anat_fn, anat_fn)
    gunzip(anat_fn)
    
    % For each echo, copy and unzip the functional timeseries image
    for i = 1:Ne
        raw_func_fn = [raw_data_dir filesep 'sub-' sprintf('%02d',subj) '_func_sub-' sprintf('%02d',subj) '_task-rest_run-01_echo-' num2str(i) '_bold.nii.gz'];
        func_fn = [func_dir filesep 'sub-' sprintf('%02d',subj) '_task-rest_run-01_echo-' num2str(i) '_bold.nii.gz'];
        copyfile(raw_func_fn, func_fn)
        gunzip(func_fn)
    end
    
end

%%

%--------------------------------------------------------------------------
% MAIN PER-SUBJECT SCRIPT
%--------------------------------------------------------------------------

% The following is done for each subject:
% (1)   Pre-real-time processing: this includes anatomical-functional
%       coregistration, tissue segmentation, masking, calculation of
%       pre-real-time tSNR, estimation of pre-real-time T2*.
% (2)   Real-time processing: each volume in the functional time-series is
%       realigned (all echoes), T2* is estimated per volume, and echoes are
%       combined per volume using the three weighting schemes described in
%       the poster. 
% (3)   Post-processing: for each of the 3 timeseries resulting from the
%       real-time combination of multiple echoes using the 3 different
%       weighting schemes presented in the poster, the tSNR is calculated.
%       tSNR is also calculated for the reference echo timeseries, in order
%       to serve as the comparison of multi-echo combination to 'standard'
%       single echo EPI. These tSNR images are also warped to MNI space in
%       order to facilitate group level analyses.

for subj = 1:Nsub
    
    % close all figure windows from previous iteration
    close all

    % Initialize directory and file names for subject and related images
    subj_dir = ['sub-' sprintf('%02d', subj)];
    disp(subj_dir);
    anat_dir = [data_dir filesep subj_dir filesep 'anat'];
    func_dir = [data_dir filesep subj_dir filesep 'func'];
    s_fn = [anat_dir filesep subj_dir '_T1W.nii'];
    f_me1_fn = [func_dir filesep subj_dir '_task-rest_run-01_echo-1_bold.nii'];
    f_me2_fn = [func_dir filesep subj_dir '_task-rest_run-01_echo-2_bold.nii'];
    f_me3_fn = [func_dir filesep subj_dir '_task-rest_run-01_echo-3_bold.nii'];
    f_me_fn = {f_me1_fn, f_me2_fn, f_me3_fn};
    
    % Navigate to subject directory in order to save all output there
    cd([data_dir filesep subj_dir]);
   
    %% Pre-real-time processing
    disp('STEP 1: PRE-REAL-TIME PROCESSING')
    % Preprocess structural and first functional images
    [d, f, e] = fileparts(s_fn);
    if exist([d filesep 'rc1' f e], 'file')
        % Preprocessing is defined to have already been done if the
        % realigned (to functional space) white matter segmentation image
        % already exists. In this case, just create standard variable names
        disp('Preproc already done, writing variable names')
        preproc_data = struct;
        % Filenames related to anatomical data and segmentations
        preproc_data.forward_transformation = [d filesep 'y_' f e];
        preproc_data.inverse_transformation = [d filesep 'iy_' f e];
        preproc_data.gm_fn = [d filesep 'c1' f e];
        preproc_data.wm_fn = [d filesep 'c2' f e];
        preproc_data.csf_fn = [d filesep 'c3' f e];
        preproc_data.bone_fn = [d filesep 'c4' f e];
        preproc_data.soft_fn = [d filesep 'c5' f e];
        preproc_data.air_fn = [d filesep 'c6' f e];
        preproc_data.rstructural_fn = [d filesep 'r' f e];
        preproc_data.rgm_fn = [d filesep 'rc1' f e];
        preproc_data.rwm_fn = [d filesep 'rc2' f e];
        preproc_data.rcsf_fn = [d filesep 'rc3' f e];
        preproc_data.rbone_fn = [d filesep 'rc4' f e];
        preproc_data.rsoft_fn = [d filesep 'rc5' f e];
        preproc_data.rair_fn = [d filesep 'rc6' f e];
        preproc_data.wrstructural_fn = [d filesep 'wr' f e];
        preproc_data.wrgm_fn = [d filesep 'wrc1' f e];
        preproc_data.wrwm_fn = [d filesep 'wrc2' f e];
        preproc_data.wrcsf_fn = [d filesep 'wrc3' f e];
        preproc_data.wrbone_fn = [d filesep 'wrc4' f e];
        preproc_data.wrsoft_fn = [d filesep 'wrc5' f e];
        preproc_data.wrair_fn = [d filesep 'wrc6' f e];
        
        f_me1_fn = [func_dir filesep 'echo-1_bold_wodummies.nii'];
        f_me2_fn = [func_dir filesep 'echo-2_bold_wodummies.nii'];
        f_me3_fn = [func_dir filesep 'echo-3_bold_wodummies.nii'];
        rf_me1_fn = [func_dir filesep 'recho-1_bold_wodummies.nii'];
        rf_me2_fn = [func_dir filesep 'recho-2_bold_wodummies.nii'];
        rf_me3_fn = [func_dir filesep 'recho-3_bold_wodummies.nii'];
        rf_me_fn = {rf_me1_fn, rf_me2_fn, rf_me3_fn};
    else
        disp('Running preprocessing...')
        % If the realigned (to functional space) white matter segmentation
        % image does not already exists, preprocessing continues.
        
        % First create new 4d NIfTI files without dummy volumes
        out_fn = cell(Ne,1);
        for echo = 1:Ne
            out_fn{echo} = ['echo-' num2str(echo) '_bold_wodummies.nii'];
            clear fns;
            for i = Nstart:Nt
                fns{i-Nskip} = [f_me_fn{echo} ',' num2str(i)];
            end
            if ~exist(out_fn{echo}, 'file')
                spm_convert3Dto4D_jsh(fns, out_fn{echo});
            end
        end
        f_me1_fn = [func_dir filesep 'echo-1_bold_wodummies.nii'];
        f_me2_fn = [func_dir filesep 'echo-2_bold_wodummies.nii'];
        f_me3_fn = [func_dir filesep 'echo-3_bold_wodummies.nii'];
        f_me_fn = {f_me1_fn, f_me2_fn, f_me3_fn};
        
        % Then realign (estimate and reslice) all to the reference volume  - see function content
        realigned_echoes = spm_realignMEv2_jsh(f_me_fn, Eref, 'MP_echo');
        rf_me_fn = realigned_echoes.ME_fn;

        % Aanatomical preproc - see function content
        preproc_data = preRtPreProcME(f_me_fn, s_fn, spm_dir);

    end

    % Template data for new niftis, derived from reference echo
    template_fn = [rf_me_fn{Eref} ',1'];
    template_spm = spm_vol(template_fn);
    
    %% Construct GM, WM and CSF masks (separately and combined)
    % Get binary 3D images for each tissue type, based on a comparison of
    % the probability value for each tissue type per voxel (after applying
    % a treshold on the probability values)
    [GM_img_bin, WM_img_bin, CSF_img_bin] = createBinarySegments(preproc_data.rgm_fn, preproc_data.rwm_fn, preproc_data.rcsf_fn, 0.5);
    % get vectors of indices per tissue type
    I_GM = find(GM_img_bin);
    I_WM = find(WM_img_bin);
    I_CSF = find(CSF_img_bin);
    % combine binary images of all tissue types to generate mask
    mask_reshaped = GM_img_bin | WM_img_bin | CSF_img_bin;
    % get vector of indices for mask
    I_mask = find(mask_reshaped);
    % Determine some descriptive variables
    Nmaskvox = numel(I_mask);
    Nvox = numel(GM_img_bin);
    [Ni, Nj, Nk] = size(GM_img_bin);
    
    %% Calculate tSNR, T2* and S0 from full multi-echo timeseries data
    % This is done in order to get an estimate of whole brain tSNR and T2*
    % (pre-real-time), which are both used for real-time weighted
    % multi-echo-combination (methods i and ii, respectively, from the
    % poster).
    disp('STEP 2: ESTIMATE tSNR, T2STAR AND S0 MAPS')
    cd([data_dir filesep subj_dir]);
    % Create empty arrays
    F = cell(Ne,1);
    F_ave2D = cell(Ne,1);
    F_ave = cell(Ne,1);
    F_tSNR2D = cell(Ne,1);
    F_tSNR = cell(Ne,1);
    Ndyn = Nt - Nskip; % Number of dynamics to use
    
    % First calculate tSNR per echo timeseries, which is the timeseries
    % mean divided by the standard deviation of the timeseries
    for e = 1:Ne
        disp(['tSNR for echo ' num2str(e)])
        F{e} = spm_read_vols(spm_vol(rf_me_fn{e}));
        F_ave2D{e} = mean(reshape(F{e},Ni*Nj*Nk, Ndyn), 2);
        F_ave{e} = reshape(F_ave2D{e}, Ni, Nj, Nk);
        F_tSNR2D{e} = F_ave2D{e}./std(reshape(F{e},Ni*Nj*Nk, Ndyn), 0, 2);
        F_tSNR{e} = reshape(F_tSNR2D{e}, Ni, Nj, Nk);
    end
    
    disp('T2star and S0 estimation')
    % Then estimate T2* and SO using log linear regression of a simplified
    % magnetic signal decay equation (see references in poster) to the data
    % derived from averaging the three echo timeseries.
    X_pre=[ones(Ne,1) -TE(:)];
    S0_pre = zeros(Nvox,1);
    T2star_pre = zeros(Nvox,1);
    T2star_pre_corrected = T2star_pre;
    S_pre = [F_ave2D{1}(I_mask, :)'; F_ave2D{2}(I_mask, :)'; F_ave2D{3}(I_mask, :)'];
    S_pre = max(S_pre,1e-11); % negative or zero signal values should not be allowed
    b_pre = X_pre\log(S_pre);
    S0_pre(I_mask,:)=exp(b_pre(1,:));
    T2star_pre(I_mask,:)=1./b_pre(2,:);
    % Now threshold the T2star values based on expected (yet broad) range
    % of values
    T2star_pre_corrected(I_mask) = T2star_pre(I_mask);
    T2star_pre_corrected((T2star_pre_corrected(:)<0)) = 0;
    T2star_pre_corrected((T2star_pre_corrected(:)>T2star_thresh)) = 0;
    % Convert the estimated and corrected parameters to 3D matrices
    T2star_pre_img = reshape(T2star_pre_corrected, Ni, Nj, Nk);
    S0_pre_img = reshape(S0_pre, Ni, Nj, Nk);
    
    % Save results to nifti images for later use
    disp('Save maps')
    save('F_tSNR.mat','F_tSNR')
    spm_createNII_jsh(template_spm, F_tSNR{1}, [data_dir filesep subj_dir filesep 'tSNR_TE1_pre.nii'], 'tSNR image')
    spm_createNII_jsh(template_spm, F_tSNR{2}, [data_dir filesep subj_dir filesep 'tSNR_TE2_pre.nii'], 'tSNR image')
    spm_createNII_jsh(template_spm, F_tSNR{3}, [data_dir filesep subj_dir filesep 'tSNR_TE3_pre.nii'], 'tSNR image')
    save('T2star_pre_img.mat','T2star_pre_img')
    spm_createNII_jsh(template_spm, T2star_pre_img, [data_dir filesep subj_dir filesep 'T2star_pre_img.nii'], 'T2star image')
    save('S0_pre_img.mat','S0_pre_img')
    spm_createNII_jsh(template_spm, S0_pre_img, [data_dir filesep subj_dir filesep 'S0_pre_img.nii'], 'S0 image')
    
    %% Real-time initialisation and data prep   
    % Create several empty arrays and set parameters for real-time use
    ref_fn = [f_me_fn{Eref} ',1'];
    fdyn_fn = cell(Ne,1);
    currentVol = cell(Ne,1);
    F_dyn_img = cell(Ne,1);
    F_dyn = cell(Ne,1);
    F_dyn_resliced = cell(Ne,1);
    F_denoised = cell(Ne,1);
    F_dyn_denoised = cell(Ne,1);
    realign_params = cell(Ne,1);
    F_dyn_resliced_masked = cell(Ne,1);
    F_dyn_resliced_masked_img = cell(Ne,1);
    F_dyn_smoothed = cell(Ne,1);
    F_dyn_smoothed_masked = cell(Ne,1);
    F_dyn_smoothed_masked_img = cell(Ne,1);
    for e = 1:Ne
        F_dyn_resliced{e} = zeros(Ni*Nj*Nk, Ndyn);
        F_dyn_denoised{e} = zeros(Ni*Nj*Nk, Ndyn);
        F_dyn_resliced_masked{e} = zeros(Ni*Nj*Nk, Ndyn);
        F_dyn_resliced_masked_img{e} = zeros(Ni,Nj,Nk, Ndyn);
        F_dyn_smoothed{e} = zeros(Ni*Nj*Nk, Ndyn);
        F_dyn_smoothed_masked{e} = zeros(Ni*Nj*Nk, Ndyn);
        F_dyn_smoothed_masked_img{e} = zeros(Ni,Nj,Nk, Ndyn);
    end
    
    % Set up parameters for real-time realignment. These parameters and
    % real-time algorithms were adapted from code used in OpenNFT:
    % see: https://github.com/OpenNFT/OpenNFT
    % see also: https://www.ncbi.nlm.nih.gov/pubmed/28645842
    flagsSpmRealign = struct('quality',.9,'fwhm',5,'sep',4,...
        'interp',4,'wrap',[0 0 0],'rtm',0,'PW','','lkp',1:6);
    flagsSpmReslice = struct('quality',.9,'fwhm',5,'sep',4,...
        'interp',4,'wrap',[0 0 0],'mask',1,'mean',0,'which', 2);
    infoVolTempl = spm_vol(ref_fn);
    imgVolTempl  = spm_read_vols(infoVolTempl);
    dimTemplMotCorr     = infoVolTempl.dim;
    matTemplMotCorr     = infoVolTempl.mat;
    dicomInfoVox   = sqrt(sum(matTemplMotCorr(1:3,1:3).^2));
    nrSkipVol = 0;
    for e = 1:Ne
        realign_params{e}.A0=[];realign_params{e}.x1=[];realign_params{e}.x2=[];
        realign_params{e}.x3=[];realign_params{e}.wt=[];realign_params{e}.deg=[];realign_params{e}.b=[];
        realign_params{e}.R(1,1).mat = matTemplMotCorr;
        realign_params{e}.R(1,1).dim = dimTemplMotCorr;
        realign_params{e}.R(1,1).Vol = imgVolTempl;
    end
    
    % More parameters and empty variables for real-time use
    X=[ones(Ne,1) -TE(:)];
    base = zeros(Nvox,Ndyn);
    S0 = zeros(Nvox,Ndyn);
    T2star = zeros(Nvox,Ndyn);
    T2star_corrected = T2star;
    T2star_img = zeros(Ni, Nj, Nk, Ndyn);
    T2star_pv_img = zeros(Ni, Nj, Nk, Ndyn);
    S0_img = zeros(Ni, Nj, Nk, Ndyn);
    S0_pv_img = zeros(Ni, Nj, Nk, Ndyn);
    S_combined_img = zeros(Ni, Nj, Nk, Ndyn);
    S_combined_img2 = zeros(Ni, Nj, Nk, Ndyn);
    T = zeros(Ndyn,1);
    S0_pv = zeros(Nvox,Ndyn);
    T2star_pv = zeros(Nvox,Ndyn);
    T2star_pv_corrected = T2star_pv;
    combined_t2s_pre = base;
    combined_tsnr_pre = base;
    combined_t2s_rt = base;
    N = [Ni, Nj, Nk];
    
    %% Real-Time processing
    % For each new volume in the functional time-series:
    % a - realign all echoes to the reference echo, 
    % b - estimate T2* using log-linear regression
    % c - combine the echoes using the three weighting schemes described in the poster. 
    for i = 1:Ndyn
        tic;
        % a - realign all echoes to the reference echo, 
        for e = 1:Ne
            fdyn_fn{e} = [f_me_fn{e} ',' num2str(i)]; % filename of dynamic functional image
            currentVol{e} = spm_vol(fdyn_fn{e});
            F_dyn_img{e} = spm_read_vols(currentVol{e}); % this is the unprocessed image
            realign_params{e}.R(2,1).mat = currentVol{e}.mat;
            realign_params{e}.R(2,1).dim = currentVol{e}.dim;
            realign_params{e}.R(2,1).Vol = F_dyn_img{e};
            
            % realign (FROM OPENNFT: preprVol.m)
            [realign_params{e}.R, realign_params{e}.A0, realign_params{e}.x1, realign_params{e}.x2, realign_params{e}.x3, realign_params{e}.wt, realign_params{e}.deg, realign_params{e}.b, realign_params{e}.nrIter] = spm_realign_rt(realign_params{e}.R, flagsSpmRealign, i, nrSkipVol + 1, realign_params{e}.A0, realign_params{e}.x1, realign_params{e}.x2, realign_params{e}.x3, realign_params{e}.wt, realign_params{e}.deg, realign_params{e}.b);
            
            % MC params (FROM OPENNFT: preprVol.m)
            tmpMCParam = spm_imatrix(realign_params{e}.R(2,1).mat / realign_params{e}.R(1,1).mat);
            if (i == nrSkipVol + 1)
                realign_params{e}.offsetMCParam = tmpMCParam(1:6);
            end
            realign_params{e}.motCorrParam(i,:) = tmpMCParam(1:6)-realign_params{e}.offsetMCParam; % STEPHAN NOTE: I changed indVolNorm to indVol due to error, not sure if this okay or wrong?
            realign_params{e}.MP(i,:) = realign_params{e}.motCorrParam(i,:);
            % Reslice (FROM OPENNFT: preprVol.m)
            realign_params{e}.reslVol = spm_reslice_rt(realign_params{e}.R, flagsSpmReslice);
            
            F_dyn_resliced{e}(:,i) = realign_params{e}.reslVol(:);
            F_dyn_resliced_masked{e}(I_mask,i) = F_dyn_resliced{e}(I_mask,i);
        end                    
        
        % b - estimate T2* using log-linear regression
        S_pv = [F_dyn_resliced{1}(I_mask,i)'; F_dyn_resliced{2}(I_mask,i)'; F_dyn_resliced{3}(I_mask,i)'];
        S_pv = max(S_pv,1e-11);  %negative or zero signal values not allowed
        b_pv = X\log(S_pv);
        S0_pv(I_mask,i)=exp(b_pv(1,:));
        T2star_pv(I_mask,i)=1./b_pv(2,:);
        if isnan(b_pv)
            disp(['isnan: i = ' i ])
        end
        % Now threshold the T2star values based on expected (yet broad) range of values
        T2star_pv_corrected(I_mask,i) = T2star_pv(I_mask,i);
        T2star_pv_corrected((T2star_pv_corrected(:,i)<0), i) = 0;
        T2star_pv_corrected((T2star_pv_corrected(:,i)>T2star_thresh), i) = 0;
        
        % c - Combine the echoes using the three weighting schemes described in the poster. 
        F = {F_dyn_resliced{1}(:,i), F_dyn_resliced{2}(:,i), F_dyn_resliced{3}(:,i)};
        % Combination method 1: pre-real-time T2*
        weight_img = T2star_pre_img;
        combined_t2s_pre(:, i) = combine_echoes_v2(TE, 1, weight_img, F, I_mask, N);
        % Combination method 2: pre-real-time tSNR
        weight_img = F_tSNR;
        combined_tsnr_pre(:, i) = combine_echoes_v2(TE, 2, weight_img, F, I_mask, N);
        % Combination method 3: real-time T2*
        weight_img = reshape(T2star_pv_corrected(:,i), Ni, Nj, Nk);
        combined_t2s_rt(:, i) = combine_echoes_v2(TE, 1, weight_img, F, I_mask, N);
        
        T(i) = toc;
        disp(['i=' num2str(i) ': ' num2str(T(i))]);
    end
    
    %% Calculate tSNR for 4 timeseries
    % 1 - 2nd echo realigned
    % 2 - combined echoes using precalculated T2star for weighting
    % 3 - combined echoes using precalculated tSNR for weighting
    % 4 - combined echoes using real-time T2star for weighting
    
    % Create empty arrays
    tSNR = cell(4,1);
    tSNR_img = cell(4,1);
    tSNR_img_fn = cell(4,1);
    for i = 1:numel(tSNR)
        tSNR{i} = zeros(Nvox,1);
    end
    
    % Calculate tSNR
    m1 = mean(F_dyn_resliced{2}(I_mask,:), 2);
    stddev1 = std(F_dyn_resliced{2}(I_mask,:), 0, 2);
    tSNR{1}(I_mask) = m1./stddev1;
    tSNR{1}(isnan(tSNR{1}))=0;
    
    m2 = mean(combined_t2s_pre(I_mask,:), 2);
    stddev2 = std(combined_t2s_pre(I_mask,:), 0, 2);
    tSNR{2}(I_mask) = m2./stddev2;
    tSNR{2}(isnan(tSNR{2}))=0;
    
    m3 = mean(combined_tsnr_pre(I_mask,:), 2);
    stddev3 = std(combined_tsnr_pre(I_mask,:), 0, 2);
    tSNR{3}(I_mask) = m3./stddev3;
    tSNR{3}(isnan(tSNR{3}))=0;
    
    m4 = mean(combined_t2s_rt(I_mask,:), 2);
    stddev4 = std(combined_t2s_rt(I_mask,:), 0, 2);
    tSNR{4}(I_mask) = m4./stddev4;
    tSNR{4}(isnan(tSNR{4}))=0;
    
    % Save nifti images of all tSNR maps for later use
    for i = 1:numel(tSNR)
        tSNR_img{i} = reshape(tSNR{i}, Ni, Nj, Nk);        
        tSNR_img_fn{i} = [data_dir filesep subj_dir filesep 'tSNR_ts' num2str(i) '_' subj_dir '.nii'];
        spm_createNII_jsh(template_spm, tSNR_img{i}, tSNR_img_fn{i}, 'tSNR image')
    end
    
   %% Normalize subject-space maps to MNI space in order to allow group-level comparison
    
    % Normalize tsnr images to MNI space.
    fns={};
    for i= 1:numel(tSNR_img)
        fns{i} = [data_dir filesep subj_dir filesep 'tSNR_ts' num2str(i) '_' subj_dir '.nii'];
    end
    [d1,f1,e1] = fileparts(s_fn);
    for j = (i+1):(i+6)
        fns{j} = [d1 filesep 'rc' num2str(j-i) f1 e1];
    end
    fns{j+1} = preproc_data.rstructural_fn;
    
    spm_normalizeWrite_jsh(preproc_data.forward_transformation, fns);

end


%%

%--------------------------------------------------------------------------
% GROUP-LEVEL STEPS
%--------------------------------------------------------------------------

ISMRMBenelux2019poster_ResultsScript;
