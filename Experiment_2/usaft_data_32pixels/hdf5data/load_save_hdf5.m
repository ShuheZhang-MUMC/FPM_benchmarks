clc
clear

freqXY_calib = h5read("Image_sections_color_0.hdf5","/section0/positions");
I_low = h5read("Image_sections_color_0.hdf5","/section0/ptychogram");
LED_pos = h5read("Image_sections_color_0.hdf5","/section0/positions_led");



lambda = 0.622;
NA = 0.07;

pix = 32;
fx = (-pix/2:pix/2-1) / (3.45 * pix / 1.53);
df = fx(2) - fx(1);

saved_data.lambda = lambda;
saved_data.I_low = I_low;
saved_data.na_cal = NA;
saved_data.mag = 1.6;
saved_data.dpix_c = 3.45;
saved_data.na_rp_cal = (NA / lambda / df);
saved_data.freqXY_calib = freqXY_calib' - 2;
% saved_data.na_calib = na_calib;

save('saved_data_test.mat','saved_data')
run_benchmarks;
