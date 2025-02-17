function [loss,dm] = retinex_loss(x,y,k,low_fil)
% calculation retinex loss for surrounding retinex theory

% x: Targets
% y: Predicts

filt = @(o) imfilter(o,low_fil,'circular');
k = k^2;

% forward of surrounded retinex
fil_x   = filt(x);
fil_y   = filt(y);

m11 = bsxfun(@times,fil_x, fil_x);
m22 = bsxfun(@times,fil_y, fil_y);
mxx = bsxfun(@times,fil_x, fil_y);

sig1 = filt(x.^2) - m11;
sig2 = filt(y.^2) - m22;
sigX = filt(x.*y) - mxx;


cs_p = (2*sigX + k) ./ (sig1 + sig2 + k);
cs_x = 2./(sig1 + sig2 + k);

% backward of surrounded retinex
c1 =  filt(cs_x) .* y;
c2 = -filt(cs_x .* fil_y);
c3 = -filt(cs_x .* cs_p) .* x;
c4 =  filt(cs_x .* cs_p.*fil_x);

dm = (c1 + c2 + c3 + c4);

ret_map = bsxfun(@rdivide,2 * sigX + k,sig1 + sig2 + k);
loss = 1 - mean(ret_map(:));

end