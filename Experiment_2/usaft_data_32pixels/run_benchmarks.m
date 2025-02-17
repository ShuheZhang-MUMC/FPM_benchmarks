clc
clear




addpath(genpath('func_ware'));
addpath(genpath('solvers'))

filename = "saved_data_test";
path0 = "hdf5data/" + filename + ".mat";

path_save = 'outputs/';

load(path0);

if_show = true;

func1 = @(x) EPRY_recovery_mFPM(x,if_show);
func2 = @(x) EPRY_recovery_AS(x,if_show);
func3 = @(x) EPRY_recovery_ADMM(x,if_show);
func4 = @(x) FD_FPM_recovery(x,if_show);
func5 = @(x) VEM_FPM_recovery(x,if_show);
func6 = @(x) APIC_recovery(x);

func_all = {func1,func2,func3,func4,func5,func6};

out = zeros(32*8,32*8,6);
dur = zeros(6,1);
for num_test = 1:5
    [out(:,:,num_test),dur(num_test,1)] = func_all{num_test}(saved_data);
end
% save(['algo',num2str(num_test),'.mat'],'out');

save_data(path_save,filename,out,dur,out)


%% saving data
function save_data(path_save,filename,snr_data,dur,out)
    save([path_save,'snr_',filename(1:end-4),'.mat'],'snr_data','dur','out')
end