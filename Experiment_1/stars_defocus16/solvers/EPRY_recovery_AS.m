function [out,durations] = EPRY_recovery_AS(loaded_data,if_show)
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

% S = imresize(sqrt(I_camera(:,:,1)),[PIX,PIX])/ m_factor^0;
S = imresize(sqrt(mean(I_camera(:,:,1:9),3)),m_factor);

objectFT = fftshift(fft2(S));
pupil = 1 ;


freqXY = loaded_data.freqXY_calib;
fxc0 = round(freqXY(:,2) - pix_CCD/2 + PIX/2);
fyc0 = round(freqXY(:,1) - pix_CCD/2 + PIX/2);

for iter = 1:120
    if iter > 2
        if step_size == 0
            break;
        end
    end
    for con = 1:number_img
        if iter == 1 && con == 1
            step_size = 1;
            error_bef = inf;
        elseif con == 1 && iter >3
            error_now = 0;
            for leds = 1:number_img
                fxc = fxc0(con,1);
                fyc = fyc0(con,1);
                fxl=round(fxc-(pix_CCD-1)/2);fxh=round(fxc+(pix_CCD-1)/2);
                fyl=round(fyc-(pix_CCD-1)/2);fyh=round(fyc+(pix_CCD-1)/2);
                CTF_system = pupil.*CTF_object0;   
                F_sub_old = objectFT(fyl:fyh,fxl:fxh) .* CTF_system;  
                img_cal = ifft2(ifftshift(F_sub_old)) / m_factor^2;
                diff_img = (abs(img_cal) - sqrt(abs(I_camera(:,:,leds)))).^2;
                error_now = error_now + sum(diff_img(:));
            end  
            disp(['at ',num2str(iter),'-th iter, res = ',...
                        num2str(error_now),', eta =',...
                        num2str(step_size)]);
            
            if((error_bef - error_now)/error_bef<0.001)
                % Reduce the stepsize when no sufficient progress is made
                step_size = step_size/2;
                % Stop the iteration when Alpha is less than 0.001(convergenced)
                if(step_size<0.0001)
                    step_size = 0;
                end
            end
            error_bef = error_now;
        end
                    
        fxc = fxc0(con,1);
        fyc = fyc0(con,1);
        fxl=round(fxc-(pix_CCD-1)/2);fxh=round(fxc+(pix_CCD-1)/2);
        fyl=round(fyc-(pix_CCD-1)/2);fyh=round(fyc+(pix_CCD-1)/2);

        CTF_system = pupil.*CTF_object0;   

        oj = objectFT(fyl:fyh,fxl:fxh);

        forward =  ifft2(ifftshift(oj .* CTF_system))  / m_factor^2;
        
        dm = sqrt(abs(I_camera(:,:,con))) - abs(forward);

        forward = fftshift(fft2(dm .* sign(forward).* m_factor^2));


        objectFT(fyl:fyh,fxl:fxh) = objectFT(fyl:fyh,fxl:fxh) + ...
                       step_size * deconvolution_pie(forward,CTF_system,'ePIE');
        pupil = pupil +...
                step_size * deconvolution_pie(forward,oj,'ePIE');
    end
    
    img = ifft2(ifftshift(objectFT));
    ic = abs(img);
    
    if if_show
        figure(2022);
        subplot(1,2,1)
        imshow(angle(pupil),[]);
        title('Fourier spectrum');

        % Show the reconstructed amplitude
        subplot(1,2,2)
        imshow(ic,[]);
        title(['Iteration No. = ',int2str(iter)]);
        drawnow;
    end
end
% imwrite(uint8(255*mat2gray(mod(angle(pupil.*CTF_object0),2*pi))),twilight_shifted(256),'aberra_AS.png')
% 
% img = ifft2(ifftshift(objectFT));
% imwrite(mat2gray(abs(img)),'aberra_AS_I.png')
durations = toc(timer_start);
out = objectFT;
% img = ifft2(ifftshift(objectFT));
% figure();imshow(abs(img),[]);title('amplitude, AS method')
% figure();imshow(angle(img),[]);title('phase, AS method')


end

function out = deconvolution_pie(in,ker,type)
    switch type
        case 'ePIE' % Fourier space deconvolution using ePIE type weight
            out = conj(ker) .* in ./ max(max(abs(ker).^2));
        case 'tPIE' % Fourier space deconvolution using tPIE type weight
            fenzi = conj(ker) .* in;
            fenmu = (abs(ker).^2 + 1e-5);

            bias = abs(ker) ./ max(abs(ker(:)));

            out = bias .* fenzi ./ fenmu;
        case 'rPIE' % Fourier space deconvolution using rPIE type weight
            r1 = 0.999;
            fenzi = conj(ker) .* in;
            fenmu = r1 * abs(ker).^2 + (1 - r1) * max(max(abs(ker).^2));
            
            bias = abs(ker) ./ max(abs(ker(:)));
            out = bias .* fenzi ./ fenmu;
        case 'none' 
            out = conj(ker) .* in;
        otherwise 
            error('type (#3 parameter) must be rPIE, ePIE or tPIE')
    end
end

function CTF = get_CTF(pix_CCD,pix_sz,mag,NA,lambda)
    fx = (-pix_CCD/2:pix_CCD/2-1)/(pix_sz * pix_CCD / mag);
    fy = fx;
    [fx,fy] = meshgrid(fx,fy);
    CTF = double(abs(fx + 1i*fy) < (NA/lambda));
end