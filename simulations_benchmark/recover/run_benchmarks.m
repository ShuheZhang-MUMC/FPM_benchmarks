clc
clear

addpath(genpath('solvers'));

path0 = 'noise_test'; % loading path, should be adjusted
path1 = '';

ground_truth = load('ground_truth.mat');
I = ground_truth.ground_truth_u;


addpath(genpath('func_ware'));
index = dir([path0,'\',path1]);
path_save = 'saved_score\';

for ccccc = 3:length(index)


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
for num_test = 6
    [out(:,:,num_test),dur(num_test,1)] = func_all{num_test}(saved_data);
end

clc
snr_data = zeros(2,6);
snr_data_norm = zeros(2,6);
for con = 1:6
obj = gather(ifft2(ifftshift(out(:,:,con))));

% register results
obj = abs(obj);
[output, Greg] = dftregistration(fft2(I),fft2(obj),100); 
obj = abs((ifft2(Greg)));

% normalize intensity
obj = mat2gray(obj);  
I_ssim = ssim(obj,I);
I_psnr = psnr(obj,I);
snr_data(1,con) = I_psnr;
snr_data(2,con) = I_ssim;
disp(['the psnr for ',num2str(con),'-method is ',num2str(I_psnr)]);
disp(['the ssim for ',num2str(con),'-method is ',num2str(I_ssim)]);
end

save_data(path_save,filename,snr_data,snr_data_norm,dur)
end


function save_data(path_save,filename,snr_data,snr_data_norm,dur)
    save([path_save,'snr_',filename(1:end-4),'.mat'],'snr_data','snr_data_norm','dur')
end
