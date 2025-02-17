clc
clear

load('snr_Siemens_defocus16_reduced_2.mat');
snr_data = out;
img = [];
img_phase = [];

img_enlarge = [];

blanc = ones(2048,100);
for con = 1:6
    this_img = snr_data(:,:,con);
    img_temp = ifft2(ifftshift(this_img));
    img_real = abs(img_temp);
    img_real(1,1) = 0;
    img_real = mat2gray(img_real);
   
    img_ang = mat2gray(angle(img_temp));
    this_enlarge = imresize(img_real(1024-256:1024+256-1,1024-256:1024+256-1),[2048,2048],"box");

    if con < 6
        img = [img,img_real,blanc];
        img_enlarge = [img_enlarge,this_enlarge,blanc];

        
    else
        img = [img,img_real];
        img_enlarge = [img_enlarge,this_enlarge];
    end
end
figure();imshow([img;ones(100,size(img,2));img_enlarge],[])

img_all = [img;ones(100,size(img,2));img_enlarge];
imwrite(img_all,'test_N1_S1.png')