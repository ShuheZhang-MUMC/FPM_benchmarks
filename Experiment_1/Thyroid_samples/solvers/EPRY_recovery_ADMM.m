function [out,durations] = EPRY_recovery_ADMM(loaded_data,if_show)

% init datas
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

% S = imresize(sqrt(mean(I_camera(:,:,1:9),3)),[PIX,PIX])/ m_factor^0;
S = imresize(sqrt(mean(I_camera(:,:,2:9),3)),m_factor);
objectFT = fftshift(fft2(S));
pupil = 1 ;
Alpha = 1;
pupil_history = zeros(pix_CCD,pix_CCD,30);

I_camera = sqrt(abs(I_camera));



freqXY = loaded_data.freqXY_calib;
fxc0 = round(freqXY(:,2) - pix_CCD/2 + PIX/2);
fyc0 = round(freqXY(:,1) - pix_CCD/2 + PIX/2);

q_para = gpuArray(zeros(pix_CCD,pix_CCD,number_img));
p_para = gpuArray(zeros(pix_CCD,pix_CCD,number_img));
w_para = gpuArray(zeros(pix_CCD,pix_CCD,number_img));
m_para = gpuArray(zeros(PIX,PIX));
s_para = gpuArray(zeros(PIX,PIX));

para_alpha = 1;
para_gamma = 1;
para_delta = 1;
para_eta = 1;
for loop = 1:120
    disp(num2str(loop))
    s_para = s_para * 0;
    for con = 1:number_img
        fxc = fxc0(con,1);
        fyc = fyc0(con,1);
        
        fxl=round(fxc-(pix_CCD-1)/2);fxh=round(fxc+(pix_CCD-1)/2);
        fyl=round(fyc-(pix_CCD-1)/2);fyh=round(fyc+(pix_CCD-1)/2);
        
        CTF_system = pupil.*CTF_object0;
    
        F_sub_old = objectFT(fyl:fyh,fxl:fxh) .* CTF_system;
        if loop ==1
            m_para(fyl:fyh,fxl:fxh) = m_para(fyl:fyh,fxl:fxh) + abs(CTF_system).^2;
        end
        
        temp_q = F_sub_old - w_para(:,:,con);
        img_old = ifft2(ifftshift(temp_q))/m_factor^2;
        
        q_para(:,:,con) = temp_q + 1./(1 + para_alpha).*...
              fftshift(fft2((I_camera(:,:,con)).*sign(img_old) - img_old)) * m_factor^2;
          
        s_para(fyl:fyh,fxl:fxh) = s_para(fyl:fyh,fxl:fxh) + ...
            (q_para(:,:,con) + w_para(:,:,con)).*conj(CTF_system);
        
    end

    fenzi = para_gamma * (para_delta/para_alpha) + s_para;
    fenmu = para_gamma * (para_delta/para_alpha) + m_para;
    objectFT = fenzi./(fenmu + eps);
    
    if if_show
        figure(2022)
        subplot(1,2,1)
        imshow(log(abs((objectFT)+1)),[0, max(max(log(abs((objectFT)+1))))/2]);
        title('Fourier spectrum');

        % Show the reconstructed amplitude
        subplot(1,2,2)
        imshow(abs(ifft2(ifftshift(objectFT))),[]);
        title(['Iteration No. = ',int2str(loop)]);
        drawnow;
    end
    
    for con = 1:number_img
        fxc = fxc0(con,1);
        fyc = fyc0(con,1);

        fxl=round(fxc-(pix_CCD-1)/2);fxh=round(fxc+(pix_CCD-1)/2);
        fyl=round(fyc-(pix_CCD-1)/2);fyh=round(fyc+(pix_CCD-1)/2);
        
        CTF_system = pupil.*CTF_object0;%conv2(pupil.*CTF_object0,fspecial('gaussian',15,6),'same');   
    
        F_sub_old = objectFT(fyl:fyh,fxl:fxh) .* CTF_system;
        
        w_para(:,:,con) = w_para(:,:,con) + para_eta.*...
            (q_para(:,:,con) - F_sub_old);
    end
    
end

out = objectFT;
durations = toc(timer_start);
end

function CTF = get_CTF(pix_CCD,pix_sz,mag,NA,lambda)
    fx = (-pix_CCD/2:pix_CCD/2-1)/(pix_sz * pix_CCD / mag);
    fy = fx;
    [fx,fy] = meshgrid(fx,fy);
    CTF = double(abs(fx + 1i*fy) < (NA/lambda));
end

function out = CTTV_filter_ATV(img,lambda,iter)
[m,n,~] = size(img);


otf_dx = psf2otf([-1,1],[m,n]);
otf_dy = psf2otf([-1;1],[m,n]);

DTD = abs(otf_dx).^2 + abs(otf_dy).^2;



o = img;
fft_s = fft2(img);

lambda0 = lambda;
lambda_max = 1e5;

fft_o = fft2(o);

% for loop = 1:iter
while lambda0 < lambda_max
    % g sub-problem 
    gx0 = real(ifft2(fft_o .* otf_dx));
    gy0 = real(ifft2(fft_o .* otf_dy));
%     ss = gx0.^2 + gy0.^2;
    
%     gx = gx0 .* (gx0.^2 > lambda/lambda0);
%     gy = gy0 .* (gy0.^2 > lambda/lambda0);
    gx = sign(gx0).* max(abs(gx0) - lambda/lambda0,0);
    gy = sign(gy0).* max(abs(gy0) - lambda/lambda0,0);
    
    
    % o sub-problem
    fenzi = fft_s + lambda0 * (conj(otf_dx).*(fft2(gx)) +...
                               conj(otf_dy).*(fft2(gy)));
    fenmu = 1 + lambda0 * DTD;
    fft_o = fenzi./fenmu;
    o = real(ifft2(fft_o));
    
    lambda0 = lambda0 * 2;
end

out = o;
end

