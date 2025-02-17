clc
clear

path0 = '';
path1 = 'test_data';

addpath(genpath('func_ware'));
addpath(genpath('solvers'));

index = dir(path1);
path_save = '';

% for ccccc = 225:length(index)

% data_name = ['snr_Lseq_LED=',num2str(label(1)),...
%               '___noise_amplitude=',num2str(label(2)),...
%               '___illu_c=',num2str(label(3)),...
%               '___group=',num2str(ccccc),'.mat'];

filename = index(3).name;
saved_data = load([path1,'\',filename]);

temp = saved_data.freqXY_calib;
temp = fliplr(temp);
saved_data.freqXY_calib = temp;
if_show = true;

func1 = @(x) EPRY_recovery_mFPM(x,if_show);
func2 = @(x) EPRY_recovery_AS(x,if_show);
func3 = @(x) EPRY_recovery_ADMM(x,if_show);
func4 = @(x) FD_FPM_recovery(x,if_show);
func5 = @(x) VEM_FPM_recovery(x,if_show);
func6 = @(x) APIC_recovery(x);

func_all = {func1,func2,func3,func4,func5,func6};

out = zeros(512*4,512*4,6);
dur = zeros(6,1);
for num_test = 4
    [out(:,:,num_test),dur(num_test,1)] = func_all{num_test}(saved_data);
end

% end
clc

save_data(path_save,filename,out,dur)



function save_data(path_save,filename,snr_data,dur)
    save([path_save,'snr_',filename(1:end-4),'.mat'],'snr_data','dur')
end