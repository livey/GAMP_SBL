function re_x = GAMPSBL_m1(A,y,maxiter,tor)
%%
 % for beta=0 and noisy case: C=1e0;
 % for beta~=0 case; C=1e6;
%%
beta=0;
C=1e0;
% initialization
A_t          =  A';
A2           =  A.*A;
A_t2         =  A_t.*A_t;
[M,N]        =  size(A);
Thresholding =  1.0e-6;
time         =  0;
% maxiter      =  200;
x_prior      =  (A_t*y)./diag(A_t*A);  % in high dimensional case. replaced as At*y since A_t*A is highly computing cost
% x_prior  = inv(A'*A)*A'*y;
% sigma2     =  norm(y)^2/((1+C)*M); % optional
% sigma2       =  std(y)^2/C; 
sigma2=1e-10;
% x_prior      =  ones(N,1);         % optional
x_var_prior  =  ones(N,1);
s_prior      =  zeros(M,1); 
alpha        =  ones(N,1);
% beta         =  1;
a            =  1;
% adaptive a 
a = 1-0.2*M/N;

b            =  1e-15;
x_hat        =  zeros(N,1);

% iterative process
while time < maxiter && norm(x_prior-x_hat) > Thresholding
    time   =  time + 1;
    x_hat  =  x_prior;
   for jj=1:10
    
    % factor update part
    z      =  A*x_prior; 
    tao_p  =  A2*x_var_prior;
    p_hat  =  z-tao_p.*s_prior; 
    
    tao_z  =  (tao_p*sigma2)./(tao_p+sigma2);
    z_hat  =  tao_z.*(y./sigma2+p_hat./tao_p);
    s_prior  =  (z_hat-p_hat)./tao_p;
    tao_s  =  (1-tao_z./tao_p)./tao_p;
       
    % geting the limit 
%     tao_z = 0;
%     z_hat = y;
%     s_prior  =  (z_hat-p_hat)./tao_p;
%     tao_s = 1./tao_p;
    
    % variable update part
    tao_l  =  (A_t2*tao_s);   %  tao_r = 1./(A_t.*A_t*tao_s); tao_r =1./tao_l;
    tao_r  =   1./tao_l;
    r_hat  =  x_prior+tao_r.*(A_t*s_prior);
    
    lalpha =  [alpha(2:N);0]; 
    ralpha =  [0;alpha(1:N-1)]; 
    k_alpha = alpha +beta*lalpha+beta*ralpha;
    x_prior = (r_hat.*tao_l)./(k_alpha+tao_l);
    x_var_prior = 1./(k_alpha+tao_l);
   end  
    % EM (parameters update)
    w=x_prior.^2+x_var_prior;
    alpha = (a-.5)./(0.5*w+b);
    index=find(alpha>1e10);  alpha(index)=1e10;
    
end

re_x = x_prior;
