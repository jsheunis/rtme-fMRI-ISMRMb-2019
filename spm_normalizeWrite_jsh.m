function spm_normalizeWrite_jsh(deformation, filenames)

normalize_write = struct;
% Subject
% Deformation field
normalize_write.matlabbatch{1}.spm.spatial.normalise.write.subj.def = {deformation};
% Data
normalize_write.matlabbatch{1}.spm.spatial.normalise.write.subj.resample = filenames';
% Write options
normalize_write.matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = [-78,-112,-70;78,76,85];
normalize_write.matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [2,2,2];
normalize_write.matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;
normalize_write.matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = 'w';

cfg_util('run',normalize_write.matlabbatch);
disp('done')
