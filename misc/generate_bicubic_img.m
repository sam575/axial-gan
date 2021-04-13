function generate_bicubic_img()
%% Ref: https://github.com/xinntao/BasicSR
%% matlab code to genetate mod images, bicubic-downsampled images and
%% bicubic_upsampled images
clear all
%% set configurations

% comment the unnecessary lines
input_folder = '../../dataset/Odin3/cropped_images';
% saves images with LR size
% save_lr_folder = '../../dataset/Odin3/16_bicubic';
% save images with org_size upsampled from LR size
save_bic_folder = '../../dataset/Odin3/24_rec';

% up_scale = 16;
lr_size = [24 24]; % LR size
org_size = [128 128]; % Upsampling size

if exist('save_lr_folder', 'var')
    if exist(save_lr_folder, 'dir')
        disp(['It will cover ', save_lr_folder]);
    else
        mkdir(save_lr_folder);
    end
end
if exist('save_bic_folder', 'var')
    if exist(save_bic_folder, 'dir')
        disp(['It will cover ', save_bic_folder]);
    else
        mkdir(save_bic_folder);
    end
end

idx = 0;
filepaths = dir(fullfile(input_folder,'**/*.*'));
for i = 1 : length(filepaths)
    [paths, img_name, ext] = fileparts(filepaths(i).name);
    if isempty(img_name)
        disp('Ignore . folder.');
    elseif strcmp(img_name, '.')
        disp('Ignore .. folder.');
    else
        idx = idx + 1;
        str_result = sprintf('%d\t%s.\n', idx, img_name);
        fprintf(str_result);

        % read image
        img = imread(fullfile(filepaths(i).folder, [img_name, ext]));
        img = im2double(img);

        % LR
%         im_lr = imresize(img, 1/up_scale, 'bicubic');
        im_lr = imresize(img, lr_size, 'bicubic');
        if exist('save_lr_folder', 'var')
            [~, inp] = fileparts(input_folder);
            [~, out] = fileparts(save_lr_folder);
            save_folder = strrep(filepaths(i).folder, inp, out);
            if ~exist(save_folder, 'dir')
                mkdir(save_folder)
            end
            imwrite(im_lr, fullfile(save_folder, [img_name, '.png']));
        end

        % Bicubic
        if exist('save_bic_folder', 'var')
%             im_bicubic = imresize(im_lr, up_scale, 'bicubic');
            im_bicubic = imresize(im_lr, org_size, 'bicubic');
            [~, inp] = fileparts(input_folder);
            [~, out] = fileparts(save_bic_folder);
            save_folder = strrep(filepaths(i).folder, inp, out);
            if ~exist(save_folder, 'dir')
                mkdir(save_folder)
            end
            imwrite(im_bicubic, fullfile(save_folder, [img_name, '.png']));
        end
    end
end
end
