function b = get_rSNR(in,ref)


b = 0;
mom1 = 0;
mom2 = 0;
eta = 1;

r1 = 0.9;
r2 = 0.99;


for con = 1:260
    grad_b = 2*sum(sum(in + b - ref));
    mom1 = r1 * mom1 + (1 - r1) * grad_b;
    mom2 = r2 * mom2 + (1 - r2) * abs(grad_b).^2;
    
    m1 = mom1 / (1 - r1^con);
    m2 = mom2 / (1 - r2^con);
    
    grad_new = sqrt(eta) * m1./(sqrt(m2) + eps);
    b = b - grad_new;
    
    %eta = r1 * eta + (1 - r1) * abs(grad_new).^2;
end



end