function [out,durations] = EPRY_recovery_mFPM(loaded_data,if_show)
timer_start = tic;
I_camera = loaded_data.I_low;
I_camera = gpuArray(I_camera);

pratio = 4;


pix_CCD = size(I_camera,1);
PIX = pratio * pix_CCD;

number_img = size(I_camera,3);


m_factor = (PIX/pix_CCD);

CTF_object0 = get_CTF(pix_CCD,loaded_data.dpix_c,...
                              loaded_data.mag,...
                              loaded_data.na_cal,...
                              0.62863);

% S = imresize(sqrt(I_camera(:,:,1)),[PIX,PIX])/ m_factor^0;
S = imresize(sqrt(mean(I_camera(:,:,1:9),3)),m_factor);
himFT = gpuArray(fftshift(fft2(S)));

pratio = m_factor;

iteration_max = 40;
number_img = size(I_camera,3);

freqXY = loaded_data.freqXY_calib;
fxc0 = round(freqXY(:,2) - pix_CCD/2 + PIX/2);
fyc0 = round(freqXY(:,1) - pix_CCD/2 + PIX/2);

fmaskpro = double(CTF_object0);
vobj0 = zeros(PIX,PIX);
vp0 = zeros(pix_CCD, pix_CCD);
ObjT = himFT; 
PT = fmaskpro;
countimg = 0;
tt = ones(1,iteration_max*number_img);
imseqlow = gpuArray(sqrt(abs(I_camera)));

%% parameters
eta_obj = 0.2;
eta_p = 0.2;
T = 1;
alpha = 1;
beta = 1;
gamma_obj = 1;
gamma_p = 1;


for iter = 1:iteration_max
    for con = 1:number_img
        if iter == 1 && con == 1
            step_size = 1;
            error_bef = inf;
        elseif con == 1 && iter >1
            error_now = 0;
            for leds = 1:number_img
                fxc = fxc0(con,1);
                fyc = fyc0(con,1);
                fxl=round(fxc-(pix_CCD-1)/2);fxh=round(fxc+(pix_CCD-1)/2);
                fyl=round(fyc-(pix_CCD-1)/2);fyh=round(fyc+(pix_CCD-1)/2);
                CTF_system = fmaskpro .* CTF_object0;   
                F_sub_old = himFT(fyl:fyh,fxl:fxh) .* CTF_system;  
                img_cal = ifft2(ifftshift(F_sub_old)) / m_factor^2;
                diff_img = (abs(img_cal) - imseqlow(:,:,leds)).^2;
                error_now = error_now + sum(diff_img(:));
            end  
            disp(['at ',num2str(iter),'-th iter, res = ',...
                        num2str(error_now),', eta =',...
                        num2str(gamma_obj)]);
            
            error_bef = error_now;
        end
        countimg=countimg+1; 
        fxc = fxc0(con,1);
        fyc = fyc0(con,1);
        fxl=round(fxc-(pix_CCD-1)/2);fxh=round(fxc+(pix_CCD-1)/2);
        fyl=round(fyc-(pix_CCD-1)/2);fyh=round(fyc+(pix_CCD-1)/2);
        O_j=himFT(fyl:fyh,fxl:fxh); 
        CTF_system = fmaskpro .* CTF_object0;   

        lowFT = O_j .* CTF_system;
        im_lowFT = ifft2(ifftshift(lowFT));
%         tt(1,con+(iter-1)*numim)=(mean(mean(abs(im_lowFT)))/mean(mean(pratio^2 * abs(imseqlow(:,:,con))))); % LED intensity correctioin 
%         if iter>2
%             imseqlow(:,:,con)=imseqlow(:,:,con) .* tt(1,con+(iter-1)*numim); 
%         end     

        updatetemp = pratio^2 * imseqlow(:,:,con);
        im_lowFT=updatetemp.*exp(1j.*angle(im_lowFT)); 
        lowFT_p=fftshift(fft2(im_lowFT));

        himFT(fyl:fyh,fxl:fxh)=himFT(fyl:fyh,fxl:fxh)+...
        gamma_obj.*conj(CTF_system).*((lowFT_p-lowFT))./((1-alpha).*abs(CTF_system).^2 + alpha.*max(max(abs(CTF_system).^2)));
    
        fmaskpro=fmaskpro+gamma_p.*conj(O_j).*((lowFT_p-lowFT))./((1-beta).*abs(O_j).^2 + beta.*max(max(abs(O_j).^2)));            
        
        if countimg == T % momentum method
            vobj = eta_obj.*vobj0 + (himFT - ObjT);
            himFT = ObjT + vobj;
            vobj0 = vobj;                  
            ObjT = himFT;

%             vp = eta_p.*vp0 + (fmaskpro - PT);
%             fmaskpro = PT + vp;
%             vp0 = vp;
%             PT = fmaskpro;

            countimg = 0;
        end
    end
    if iter > 68
        gamma_obj = gamma_obj * 0.5;
        gamma_p = gamma_p * 0.5;
    end
    objectFT = himFT;
    
    if if_show
        figure(2022)
        subplot(1,2,1)
        imshow(log(abs(objectFT) + 1),[]);
        title('Fourier spectrum');

        % Show the reconstructed amplitude
        subplot(1,2,2)
        imshow(abs(ifft2(ifftshift(objectFT))),[]);
        title(['Iteration No. = ',int2str(iter)]);
        drawnow;
    end
end
% imwrite(uint8(255*mat2gray(mod(angle(fmaskpro .* CTF_object0),2*pi))),twilight_shifted(256),'aberra_mFPM.png')
% 
% img = ifft2(ifftshift(objectFT));
% imwrite(mat2gray(abs(img)),'aberra_mFPM_I.png')
durations = toc(timer_start);
out = objectFT;
% img = ifft2(ifftshift(objectFT));
% figure();imshow(abs(img),[])



end

function o = update(s,p,s_old,s_new,step)
w = conj(p)/max(max(abs(p)));
d = abs(p)./(abs(p).^2+0.0001);
o = s + step * w.*d.*(s_new-s_old);
end

function CTF = get_CTF(pix_CCD,pix_sz,mag,NA,lambda)
    fx = (-pix_CCD/2:pix_CCD/2-1)/(pix_sz * pix_CCD / mag);
    fy = fx;
    [fx,fy] = meshgrid(fx,fy);
    CTF = double(abs(fx + 1i*fy) < (NA/lambda));
end