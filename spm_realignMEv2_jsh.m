function output = spm_realignMEv2_jsh(ME_fn, Nref, txt_fn)

% Get number of timepoints
func_spm = spm_vol(ME_fn{1});
Ndyn = numel(func_spm);
% Filenames to realign
fns={};

for e = 1:numel(ME_fn)
    clear fns;
    fns{1} = [ME_fn{Nref} ',1'];
    for i = 2:(Ndyn+1)
        fns{i} = [ME_fn{e} ',' num2str(i-1)];
    end
    
    % Data
    spm('defaults','fmri');
    spm_jobman('initcfg');
    realign_estimate_reslice = struct;
    realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.data={fns'};
    % Eoptions
    realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.quality = 0.9;
    realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.sep = 4;
    realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.fwhm = 5;
    realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.rtm = 0; % register to first
    realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.interp = 2;
    realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.wrap = [0 0 0];
    realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.weight = '';
    % Roptions
    realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.roptions.which = [1 0]; % images [2..n]
    realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.roptions.interp = 4;
    realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.roptions.wrap = [0 0 0];
    realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.roptions.mask = 1;
    realign_estimate_reslice.matlabbatch{1}.spm.spatial.realign.estwrite.roptions.prefix = 'r';
    % Run
    spm_jobman('run',realign_estimate_reslice.matlabbatch);
    
    
    % Save parameters
    [dir,fn,ext] = fileparts(ME_fn{Nref});
    par_fn = [dir filesep 'rp_' fn '.txt'];
    if ~exist(par_fn, 'file')
        disp('Movement parameter file not found!')
    end
    new_par_fn = [dir filesep txt_fn '_' num2str(e) '.txt'];
    copyfile(par_fn, new_par_fn)
    delete(par_fn)
end

% Output
output = struct;
output.ME_fn = cell(e,1);
for e = 1:numel(ME_fn)
    [d, f, ext] = fileparts(ME_fn{e});
    output.ME_fn{e} = [d filesep 'r' f ext];
end