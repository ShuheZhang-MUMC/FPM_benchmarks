clc
clear

load('outputs/snr_saved_data_test.mat');

img = [];
img_phase = [];

blanc = ones(32*8,10);
for con = 1:5
    this_img = out(:,:,con);
    img_temp = ifft2(ifftshift(this_img));
    img_real = fliplr(rot90(mat2gray(abs(img_temp)),2));

    img_ang = mat2gray(angle(img_temp));
    if con < 6
        img = [img,img_real,blanc];
        img_phase = [img_phase,img_ang,blanc];
    else
        img = [img,img_real];
        img_phase = [img_phase,img_ang];
    end
end
figure();imshow(img,[])
figure();imshow(img_phase,[])
imwrite(img,'test_N1_S1.png')