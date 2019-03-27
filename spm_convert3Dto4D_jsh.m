function output = spm_convert3Dto4D_jsh(fns, out_fn)

% Setup structure
spm('defaults','fmri');
spm_jobman('initcfg');
convert3dto4d = struct;
convert3dto4d.matlabbatch{1}.spm.util.cat.vols = fns';
convert3dto4d.matlabbatch{1}.spm.util.cat.name = out_fn;
convert3dto4d.matlabbatch{1}.spm.util.cat.dtype = 4;
% Run
spm_jobman('run',convert3dto4d.matlabbatch);