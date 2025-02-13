clc
clear
addpath(genpath('func_ware'));
addpath(genpath('solvers'));

path0 = 'E:\ShuheZhang@Tsinghua\EM-FPM\2025_simulations\noise_test_20250212';
path1 = '';

ground_truth = load('ground_truth.mat');
I = ground_truth.ground_truth_u;%imresize(ground_truth.I,[513,512]);


addpath(genpath('func_ware'));
%I = imresize(ground_truth.I,[512,512]);
% I2 = I + 0.134;
% b = get_rSNR(I2,I)



label = [1,1,1]
index = dir([path0,'\',path1]);
path_save = 'E:\ShuheZhang@Tsinghua\EM-FPM\2025_simulations\recovery\';

for ccccc = 225:length(index)

% data_name = ['snr_Lseq_LED=',num2str(label(1)),...
%               '___noise_amplitude=',num2str(label(2)),...
%               '___illu_c=',num2str(label(3)),...
%               '___group=',num2str(ccccc),'.mat'];

filename = index(ccccc).name;
load([path0,'\',path1,'\',filename]);

if_show = false;

func1 = @(x) EPRY_recovery_mFPM(x,if_show);
func2 = @(x) EPRY_recovery_AS(x,if_show);
func3 = @(x) EPRY_recovery_ADMM(x,if_show);
func4 = @(x) FD_FPM_recovery(x,if_show);
func5 = @(x) VEM_FPM_recovery(x,if_show);
func6 = @(x) APIC_recovery(x);

func_all = {func1,func2,func3,func4,func5,func6};

out = zeros(512,512,6);
dur = zeros(6,1);
parfor num_test = 1:6
    [out(:,:,num_test),dur(num_test,1)] = func_all{num_test}(saved_data);
end


% 
% out1 = out(:,:,1);
% out2 = out(:,:,2);
% out3 = out(:,:,3);

% save([path_save,'AS\out_',filename,'.mat'],'out3');
% save([path_save,'fdFPM\out_',filename,'.mat'],'out5');

% clc
% close all;
% clc
% 

clc
snr_data = zeros(2,6);
snr_data_norm = zeros(2,6);
for con = 1:6
obj = gather(ifft2(ifftshift(out(:,:,con))));

obj = abs(obj);
[output, Greg] = dftregistration(fft2(I),fft2(obj),100);

obj = abs((ifft2(Greg)));

b = mean(I(:)) - mean(obj(:)); % zero-frequence offset
I_ssim = ssim(obj + b,I);
I_psnr = psnr(obj + b,I);
snr_data(1,con) = I_psnr;
snr_data(2,con) = I_ssim;

disp(['the psnr for ',num2str(con),'-method is ',num2str(I_psnr)]);
disp(['the ssim for ',num2str(con),'-method is ',num2str(I_ssim)]);
end
clc

save_data(path_save,filename,snr_data,snr_data_norm,dur)
end


function save_data(path_save,filename,snr_data,snr_data_norm,dur)
    save([path_save,'snr_new_20250212\snr_',filename(1:end-4),'.mat'],'snr_data','snr_data_norm','dur')
end
