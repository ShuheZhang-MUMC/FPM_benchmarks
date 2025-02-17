function [out,durations] = VEM_FPM_recovery(loaded_data,if_show)

timer_start = tic;

I_camera = loaded_data.I_low;
I_camera = gpuArray(I_camera);

I_camera = I_camera - min(I_camera(:));
I_camera = I_camera / max(I_camera(:));

pratio = 8;


pix_CCD = size(I_camera,1);
PIX = pratio * pix_CCD;

number_img = size(I_camera,3);


m_factor = (PIX/pix_CCD);

CTF_object0 = get_CTF(pix_CCD,loaded_data.dpix_c,...
                              loaded_data.mag,...
                              loaded_data.na_cal,...
                              loaded_data.lambda);

global magg
magg = 1;

S = imresize(sqrt(mean(I_camera(:,:,1:9),3)),m_factor);
% imshow(S,[])


freqXY = loaded_data.freqXY_calib;
fxc0 = round(freqXY(:,1) + PIX/2);
fyc0 = round(freqXY(:,2) + PIX/2);


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



numEpochs = 40;
numIterationsPerEpoch  = size(I_camera,3) / batchSize;
numIterations = numEpochs * numIterationsPerEpoch;



%% The iterative recovery process for FP
disp('initializing parameters')


data_formated = @(x) gpuArray(single(x));

oI = (imresize(mean(I_camera(:,:,1:9),3),m_factor * magg)); 

dtd = abs(psf2otf([-1,1],[size(oI,1),size(oI,2)])).^2;
dtd = dtd + abs(psf2otf([-1;1],[size(oI,1),size(oI,2)])).^2;
dtd = data_formated(fftshift(dtd));

% load wavefront1_data6000
wavefront1 = data_formated(fftshift(fft2(oI)));
wavefront2 = data_formated(CTF_object0);   

deconv_data.hto = wavefront1 * 0;  
deconv_data.hth = data_formated(deconv_data.hto);
deconv_data.oth = data_formated(0);
deconv_data.oto = data_formated(0);

learning_rate = 0.03;

tv_max = 0.01;
beta0 = 2;
oI = (imresize(mean(I_camera(:,:,1),3),m_factor * magg)); 
oI = data_formated(ones(size(oI)));
% oI = (imresize(0.6*mean(imRaw_new(:,:,1:8),3) + 0.4*imRaw_new(:,:,1),m_factor)) + c; 

disp('begin solving-----')

v = 0;
u = 0;

error_bef = inf;
% load('pupil_GT.mat');

loss_data = [];
erro_data = [];

epoch = 0;
while epoch < numEpochs
    epoch = epoch + 1;

    fpm_cube.reset();

    deconv_data.hth = deconv_data.hth .* 0;
    deconv_data.hto = deconv_data.hto .* 0;
    deconv_data.oth = deconv_data.oth .* 0;
    deconv_data.oto = deconv_data.oto .* 0;
    
    iteration = 0;
    
    %% E-step: get latent image
    this_loss = 0;
    while hasdata(fpm_cube)
        iteration = iteration + 1;
        this_ratio = round(iteration/numIterationsPerEpoch * 1000)/1000;

        disp(['at ',num2str(epoch),'-epoch, loading for ---- ',num2str(this_ratio * 100),'%'])
        [leds,dY_obs] = fpm_cube.next();

        fxl=round(leds(:,1) - (pix_CCD-1)/2);fxh=round(leds(:,1) + (pix_CCD-1)/2);
        fyl=round(leds(:,2) - (pix_CCD-1)/2);fyh=round(leds(:,2) + (pix_CCD-1)/2);

        [loss,loss_M_step,deconv_data] = E_step(wavefront1, ...
                                                wavefront2 , ...
                                                [fxl,fxh,fyl,fyh],...
                                                deconv_data, ...
                                                dY_obs, ...
                                                m_factor, ...
                                                size(dY_obs,3),...
                                                learning_rate);
        this_loss = this_loss + loss;

    end
    loss_data = [loss_data,this_loss];

    %% M-step: deconvolution learning the parameters
    oR = complex_TV(oI,0.002,'isotropic');
    
    [wavefront1,oI,error_now] = M_step(deconv_data,...
                                       dtd, ...
                                       oR, ...
                                       'retinex');

    erro_data = [erro_data,loss_M_step];
    wavefront2 = deconv_data.oth./(deconv_data.oto + 1) .* CTF_object0;
    wavefront2 = min(max(abs(wavefront2),0.9),1.1) .* ...
                                               sign(wavefront2) .* CTF_object0;
    
    if epoch > 14
        learning_rate = learning_rate * 0.7;
        if learning_rate < 1e-5
            break;
        end
    end
    error_bef = error_now;
    if mod(epoch,1) == 0 && if_show
        figure(7);
        subplot(121);imshow(angle(wavefront2),[]);
        subplot(122);imshow(abs(oI),[]);
    end
end

durations = toc(timer_start);
out = wavefront1;
end

%% E-step
function [loss,loss_M_step,deconv_data] = E_step(wavefront1, ...
                                                wavefront2, ...
                                                kc,...
                                                deconv_data,...
                                                dY_obs, ...
                                                pratio, ...
                                                len,learning_rate)

global magg

sub_wavefront = zeros(size(dY_obs,1) * magg,...
                      size(dY_obs,2) * magg,...
                      size(dY_obs,3),'single');


% forward inference
for data_con = 1:len
    kt = kc(data_con,3);
    kb = kc(data_con,4);
    kl = kc(data_con,1);
    kr = kc(data_con,2);
    sub_wavefront(:,:,data_con) = wavefront1(kt:kb,kl:kr);
end
clear wavefront1;

%calculate latent variable
latent_z = ifft2_ware(bsxfun(@times,sub_wavefront,wavefront2),true) / pratio^2;
latent_old = latent_z;
[loss,dm] = ret_loss(abs(latent_z) - dY_obs,'isotropic');
latent_z = latent_z - learning_rate .* bsxfun(@times,dm,sign(latent_z));
clear dm;

loss_M_step = abs(latent_old - latent_z).^2;
loss_M_step = sum(loss_M_step(:));


latent_z = fft2_ware(latent_z, true) .* pratio^2;
for data_con = 1:len
    kt = kc(data_con,3);
    kb = kc(data_con,4);
    kl = kc(data_con,1);
    kr = kc(data_con,2);

    deconv_data.hto(kt:kb,kl:kr) = deconv_data.hto(kt:kb,kl:kr) + ...
                                latent_z(:,:,data_con) .* conj(wavefront2);

    deconv_data.hth(kt:kb,kl:kr) = deconv_data.hth(kt:kb,kl:kr) +...
                                                        abs(wavefront2).^2;
end

deconv_data.oth = deconv_data.oth + sum(...
    bsxfun(@times,latent_z,conj(sub_wavefront)),3);

deconv_data.oto = deconv_data.oto + sum(abs(sub_wavefront).^2,3);

end

%% M-step
function [out,o,err] = M_step(dec_o,dtd,oR,type)
    o_old = oR;

    % solving g-subproblem
    switch type
        case 'normal'
            fenzi = dec_o.hto + fftshift(fft2(oR));
            fenmu = dec_o.hth + 1;
        case 'retinex'
            fenzi = dec_o.hto .* dtd + fftshift(fft2(0.00001*oR)); %tv_max * (v - u) + 
            fenmu = dec_o.hth .* dtd  + 0.00001; %+ tv_max
        otherwise
            error('type must be normal or retinex')
    end
    out = bsxfun(@rdivide,fenzi,fenmu);

    o = ifft2_ware(out,true);
    N = size(o,1) * size(o,2);
    err_o = (1/sqrt(N))*(sqrt(sum(sum((o - o_old).^2))));
    err = err_o;

    
end


function CTF = get_CTF(pix_CCD,pix_sz,mag,NA,lambda)
    fx = (-pix_CCD/2:pix_CCD/2-1)/(pix_sz * pix_CCD / mag);
    fy = fx;
    [fx,fy] = meshgrid(fx,fy);
    CTF = double(abs(fx + 1i*fy) < (NA/lambda));
end