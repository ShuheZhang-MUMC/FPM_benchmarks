function [out,durations] = FD_FPM_recovery(loaded_data,if_show)
timer_start = tic;
I_camera = loaded_data.I_low;
I_camera = gpuArray(I_camera);
I_camera = I_camera - min(I_camera(:));
I_camera = I_camera / max(I_camera(:));
pratio = 4;


pix_CCD = size(I_camera,1);
PIX = pratio * pix_CCD;

number_img = size(I_camera,3);


m_factor = (PIX/pix_CCD);

CTF_object0 = get_CTF(pix_CCD,loaded_data.dpix_c,...
                              loaded_data.mag,...
                              loaded_data.na_cal,...
                              0.6323);

S = imresize(sqrt(mean(I_camera(:,:,2:9),3)),m_factor);
% imshow(S,[])


freqXY = loaded_data.freqXY_calib;
fxc0 = round(freqXY(:,2) - pix_CCD/2 + PIX/2);
fyc0 = round(freqXY(:,1) - pix_CCD/2 + PIX/2);


led_pos = [fxc0,fyc0];

I_camera = sqrt(abs(I_camera));
fpm_cube = combine(arrayDatastore(led_pos,  'IterationDimension',1),...
                   arrayDatastore(I_camera, 'IterationDimension',3));

    
batchSize = 25; 

fpm_cube = minibatchqueue(fpm_cube,...
            'MiniBatchSize',     batchSize,...
            'MiniBatchFormat',   ["",""],...
            'OutputEnvironment', {'gpu'},...
            'OutputAsDlarray',   false,...
            'OutputCast',      'single');



numEpochs = 60;
numIterationsPerEpoch  = size(I_camera,3) / batchSize;
numIterations = numEpochs * numIterationsPerEpoch;


%% parameters for optimizers
% learning_rate = 0.0006;
learning_rate = 0.006;
%% The iterative recovery process for FP

wavefront1 = gpuArray((S)); % object
wavefront2 = gpuArray((CTF_object0));                             % pupil function

% figure();imshow(wavefront1,[])
epoch = 0;
iteration = 0;
type = 'none';
clear imRaw led_pos uo vo;
c = 0;

optimizer_w1 = optimizer_RMSprop(0,0,0.999,0,false,learning_rate);
optimizer_w2 = optimizer_RMSprop(0,0,0.999,0,false,learning_rate);
% optimizer_w1 = optimizer_yogi(0,0,0.9,0.999,learning_rate);
% optimizer_w2 = optimizer_yogi(0,0,0.9,0.999,learning_rate);

while epoch < numEpochs
    epoch = epoch + 1;

    fpm_cube.reset();

    clc
    disp(['processing :',fix(num2str(100 * epoch/numEpochs)*100)/100,' %']);
    temp_loss = 0;
    while hasdata(fpm_cube)
        iteration = iteration + 1;
        [leds,dY_obs] = fpm_cube.next();
        
        fxl=round(leds(:,1) - (pix_CCD-1)/2);fxh=round(leds(:,1) + (pix_CCD-1)/2);
        fyl=round(leds(:,2) - (pix_CCD-1)/2);fyh=round(leds(:,2) + (pix_CCD-1)/2);

        %% forward propagation, gain gradient
        [loss,dldw1,dldw2] = fpm_forward(wavefront1 + c, wavefront2 , ...
                                                     [fxl,fxh,fyl,fyh], ...
                                                     dY_obs, ...
                                                     m_factor, ...
                                                     size(dY_obs,3), ...
                                                     type);
        

        wavefront1 = optimizer_w1.step(wavefront1,dldw1);
        wavefront2 = optimizer_w2.step(wavefront2,dldw2);

        wavefront2 = min(max(abs(wavefront2),0.9),1.1) .* sign(wavefront2);
        wavefront2 = wavefront2 .* CTF_object0;




    end
    if if_show
        o = (wavefront1 + c); 
        F = fftshift(fft2(o));
        img_spe = log(abs(F)+1);mm = max(max(log(abs(F)+1)))/2;
        img_spe(img_spe>mm) = mm;
        img_spe(img_spe<0) = 0;
        img_spe = mat2gray(img_spe);
        figure(2);
        subplot(121); imshow(abs(wavefront1),[]);  title('Intensity','FontSize',16);
        subplot(122); imshow(img_spe,[]);          title('Fourier','FontSize',16);
        drawnow;
    end

    if epoch > 25
        optimizer_w1.lr = optimizer_w1.lr * 0.5;
        optimizer_w2.lr = optimizer_w2.lr * 0.5;
    end
end
durations = toc(timer_start);
out = fftshift(fft2(wavefront1));
end


function CTF = get_CTF(pix_CCD,pix_sz,mag,NA,lambda)
    fx = (-pix_CCD/2:pix_CCD/2-1)/(pix_sz * pix_CCD / mag);
    fy = fx;
    [fx,fy] = meshgrid(fx,fy);
    CTF = double(abs(fx + 1i*fy) < (NA/lambda));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [loss,dldw1,dldw2] = fpm_forward(wavefront1, ...
                                                wavefront2, ...
                                                kc, ...
                                                dY_obs, ...
                                                pratio, ...
                                                len, ...
                                                type)



loss = 0;
dldw1 = 0*wavefront1;
sub_wavefront1 = dY_obs;

%% forward inference
ft_wavefront1 = fft2_ware(wavefront1,true);
for data_con = 1:len
    kt = kc(data_con,3);
    kb = kc(data_con,4);
    kl = kc(data_con,1);
    kr = kc(data_con,2);
    sub_wavefront1(:,:,data_con) = ft_wavefront1(kt:kb,kl:kr);
end

x = ifft2_ware(bsxfun(@times,sub_wavefront1,wavefront2),true) / pratio^2;

dY_est = (abs(x));
[loss,dm] = ret_loss(dY_est- dY_obs,'isotropic');
x           =   bsxfun(@times, dm, sign(x)) * pratio^2; %


%% backward propagation
x_record    =   fft2_ware(x,true);
x           =   deconv_pie(x_record,wavefront2,type);

mask = zeros(size(dldw1));
for data_con = 1:len
    kt = kc(data_con,3);
    kb = kc(data_con,4);
    kl = kc(data_con,1);
    kr = kc(data_con,2);
    dldw1(kt:kb,kl:kr) = dldw1(kt:kb,kl:kr) + x(:,:,data_con);
    mask(kt:kb,kl:kr) = mask(kt:kb,kl:kr) + (abs(wavefront2)>0);
end

dldw1 = ifft2_ware(dldw1,true);
dldw2 = sum(deconv_pie(x_record,sub_wavefront1,type),3);

end

function out = deconv_pie(in,ker,type)
    switch type
        case 'ePIE'
            out = conj(ker) .* in ./ max(max(abs(ker).^2));
        case 'tPIE'
            fenzi = conj(ker) .* in;
            fenmu = (abs(ker).^2 + 1e-5);
            out = fenzi ./ fenmu;
        case 'none'
            out = conj(ker) .* in;
        otherwise 
            error()
    end

end