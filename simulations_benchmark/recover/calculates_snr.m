clc
clear

path = ['D:\ShuheZhang@Tsinghua\papers\FPM_aberration\aberration\Recovery\snr_new_0904\'];


mean_data_ssim = [];
stid_data_ssim = [];
for amplitude = 1:50
    % mean_data = [];
    % stid_data = [];
    this_snr = [];
    for group = 1:10
                
        name = ['snr_outdata_amplitude=',num2str(amplitude),...
                 '_group=',num2str(group),'.mat.mat'];

        load([path,name]);

        this_snr = [this_snr,snr_data(2,:)'];
     end
     % mean_data = [mean_data;mean(this_snr)];
     % stid_data = [stid_data;std(this_snr)];

     mean_data_ssim = [mean_data_ssim,mean(this_snr')'];
     stid_data_ssim = [stid_data_ssim,std(this_snr')'];
end

mean_data_psnr = [];
stid_data_psnr = [];
for amplitude = 1:50
    % mean_data = [];
    % stid_data = [];
    this_snr = [];
    for group = 1:10
                
        name = ['snr_outdata_amplitude=',num2str(amplitude),...
                 '_group=',num2str(group),'.mat.mat'];

        load([path,name]);

        this_snr = [this_snr,snr_data(1,:)'];
     end
     % mean_data = [mean_data;mean(this_snr)];
     % stid_data = [stid_data;std(this_snr)];

     mean_data_psnr = [mean_data_psnr,mean(this_snr')'];
     stid_data_psnr = [stid_data_psnr,std(this_snr')'];
end

save datas2 mean_data_ssim stid_data_ssim mean_data_psnr stid_data_psnr

