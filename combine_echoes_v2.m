function combined_img2D = combine_echoes_v2(TE, method, weight_img, F, I_mask, N)

% Combine multiple single-time-point echoe volumes into one volume, based
% on various methods.

% v1    - initial version    - 17 July 2017:
%       - change if statements to switch case.
%       - add 'method' parameter that tells function which combination
%       method to use:
%           1 = T2*-weighted summation, use T2* timeseries per voxel (Poser)
%           2 = T2*-weighted summation, use average T2* per voxel (Poser)
%           3 = T2*-weighted summation, use resting blocks average T2* per voxel (Poser)
%           4 = TE-weighted summation, to be implemented


% v2:

% Method 1: Per-voxel T2star-weighted combination. This 3D T2* image could
% be calculated in various ways (e.g. T2star and S0 estimated per voxel
% from time-series average of multiple echoes; or T2star and S0 estimated
% in real-time). See poser et al for details.

% S = sum[S(TEn).w(TEn)n]
% w(TEn)n = [TEn.exp(?TEn/T2*)]/?[TEn.exp(?TEn/T2*)]
% thus, S = sum[S(TEn)].[TEn.exp(?TEn/T2*)]/?[TEn.exp(?TEn/T2*)]

% Method 2: Per-voxel tSNR-weighted combination. This requires a 3D tSNR
% image per echo. See poser et al for details.

% S = sum[S(TEn).w(TEn)n]
% w(TEn)n = [tSNR.TEn]/sum[tSNR.TEn]



Ne = numel(TE);
if numel(F) ~= Ne
    disp('The the number of echoes in the functional image (F) does not match the number of echo times')
    return;
end


% Nargs = nargin;

% if Nargs-4 ~= Ne
%     disp('Number of image inputs do not match echo time vector size')
% end

combined_img2D = zeros(N(1)*N(2)*N(3), 1);
% sum_weights = combined_img;
% weights_denom = cell(Ne,1);
% weights = cell(Ne,1);

switch method
    case 1
        T2star_img = weight_img;
        T2star_2D = reshape(T2star_img, N(1)*N(2)*N(3),1);
        weights_denom = TE.*exp(-TE./T2star_2D(I_mask)); % matrix = Nmask * Ne (each column represents denominator for specific echo)
        weights_sum = sum(weights_denom, 2); % matrix = Nmask * 1 (column represents sum of denominators for all echoes)
        weights = weights_denom./weights_sum; % matrix = Nmask * Ne (each column represents weighting vector for specific echo volume)
        
        f = [];
        for e = 1:Ne
            fe = reshape(F{e}, N(1)*N(2)*N(3),1);
            f = cat(2, f, fe(I_mask));
        end
        
        F_weighted = f.*weights;
        combined_masked = sum(F_weighted, 2);
        combined_img2D(I_mask) = combined_masked;

    case 2
        if numel(weight_img) ~= Ne
            disp('For method 2 (tSNR-weighted combination), the weight image should have the same dimension as the number of echoes')
            return;
        end
        
        f = [];
        tSNR = [];
        for e = 1:Ne
            fe = reshape(F{e}, N(1)*N(2)*N(3),1);
            f = cat(2, f, fe(I_mask));
            tSNRe = reshape(weight_img{e}, N(1)*N(2)*N(3),1);
            tSNR = cat(2, tSNR, tSNRe(I_mask));
        end
        
        % w(TEn)n = [tSNR.TEn]/sum[tSNR.TEn]
        weights_denom = TE.*tSNR; % matrix = Nmask * Ne (each column represents denominator for specific echo)
        weights_sum = sum(weights_denom, 2); % matrix = Nmask * 1 (column represents sum of denominators for all echoes)
        weights = weights_denom./weights_sum; % matrix = Nmask * Ne (each column represents weighting vector for specific echo volume)
        
        F_weighted = f.*weights;
        combined_masked = sum(F_weighted, 2);
        combined_img2D(I_mask) = combined_masked;
        
    otherwise
        disp('No implementation for this method')
end

% 
% 
% 
% img_size = size(T2star);
% Ndim = numel(img_size);
% all_images = cell(Ne,1);
% 
% task_time_course = zeros(208, 1);
% ampl = 1;
% task_time_course(17:32) = ampl;
% task_time_course(49:64) = ampl; 
% task_time_course(81:96) = ampl;
% task_time_course(113:128) = ampl;
% task_time_course(145:160) = ampl; 
% task_time_course(177:192) = ampl;
% II = find(task_time_course==0);
% 
% 
% % Initialise variables
% switch Ndim
%     case 3
%         combined_img = zeros(img_size(1)*img_size(2)*img_size(3),1);
%         sum_weights = combined_img;
%         weights = zeros(img_size(1)*img_size(2)*img_size(3), 1, Ne);
%         T2star_img = reshape(T2star,img_size(1)*img_size(2)*img_size(3),1);
%         for i = 1:Ne
%             all_images{i} = reshape(varargin{i},img_size(1)*img_size(2)*img_size(3),1);
%         end
%     case 4
%         combined_img = zeros(img_size(1)*img_size(2)*img_size(3), img_size(4));
%         switch method
%             case 1
%                 weights = zeros(img_size(1)*img_size(2)*img_size(3), img_size(4), Ne);
%                 sum_weights = combined_img;
%             case 2
%                 weights = zeros(img_size(1)*img_size(2)*img_size(3), Ne);
%                 sum_weights = zeros(img_size(1)*img_size(2)*img_size(3),1);
%             case 3
%                 % ...
%             otherwise
%                 % ...
%         end
%         T2star_img = reshape(T2star,img_size(1)*img_size(2)*img_size(3),img_size(4));
%         T2star_avg = mean(T2star_img, 2);
%         T2star_avg_blocks = mean(T2star_img(:, II'), 2);
%         for i = 1:Ne
%             all_images{i} = reshape(varargin{i},img_size(1)*img_size(2)*img_size(3),img_size(4));
%         end
%         
%     otherwise
%         % ...
% end
% 
% 
% % METHOD 1: T2*-weighted summation (with T2* timeseries)
% % METHOD FROM POSER ET AL: http://onlinelibrary.wiley.com/doi/10.1002/mrm.20900/epdf
% % S = sum[S(TEn).w(TEn)n]
% % w(TEn)n = [TEn.exp(?TEn/T2*)]/?[TEn.exp(?TEn/T2*)]
% % thus, S = sum[S(TEn)].[TEn.exp(?TEn/T2*)]/?[TEn.exp(?TEn/T2*)]
% % (as referenced by Kundu et al: http://www.sciencedirect.com/science/article/pii/S1053811917302410)
% % METHOD 2: T2*-weighted summation (with T2* averaged from timeseries)
% % METHOD 3: T2*-weighted summation (with T2* averaged from resting blocks of timeseries)
% % METHOD 4: Standard TE-weighted summation to be implemented ...
% switch method
%     case 1
%         for i = 1:Ne
%             weights(I_mask, :, i) = TE(i).*exp(-TE(i)./T2star_img(I_mask, :));
%         end
%         sum_weights(I_mask, :) = sum(weights(I_mask, :, :), 3);
%         for i = 1:Ne
%             combined_img(I_mask, :) = combined_img(I_mask, :) + all_images{i}(I_mask, :).*weights(I_mask, :, i)./sum_weights(I_mask, :);
%         end
%     case 2
%         for i = 1:Ne
%             weights(I_mask, i) = TE(i).*exp(-TE(i)./T2star_avg(I_mask));
%         end
%         sum_weights(I_mask, :) = sum(weights(I_mask, :), 2);
%         for i = 1:Ne
%             combined_img(I_mask, :) = combined_img(I_mask, :) + all_images{i}(I_mask, :).*weights(I_mask, i)./sum_weights(I_mask, :);
%         end
%     case 3
%         for i = 1:Ne
%             weights(I_mask, i) = TE(i).*exp(-TE(i)./T2star_avg_blocks(I_mask));
%         end
%         sum_weights(I_mask, :) = sum(weights(I_mask, :), 2);
%         for i = 1:Ne
%             combined_img(I_mask, :) = combined_img(I_mask, :) + all_images{i}(I_mask, :).*weights(I_mask, i)./sum_weights(I_mask, :);
%         end
%     otherwise
%         disp('Methods >=4 not developed yet...')
% end
% 
% 
% 
% 
% switch Ndim
%     case 3
%         combined_img = reshape(combined_img,img_size(1),img_size(2),img_size(3),1);
%     case 4
%         combined_img = reshape(combined_img,img_size(1),img_size(2),img_size(3),img_size(4));
%     otherwise
%         % ...
% end