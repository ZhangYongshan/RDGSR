function W = RDGSR(X, A, B, lambda)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明

iter_num = 10; %20;
[nSmp,nFea] = size(X);

Y = A*X+X*B;      % version 1

% initialize Gr and Gl
Dr = ones(nFea,1);
Dl = ones(nSmp,1);

for iter = 1:iter_num
	D_R = spdiags(Dr,0,nFea,nFea);
	D_L = spdiags(Dl,0,nSmp,nSmp);
        
	% update W
    G = (Y'*D_L*Y+lambda*D_R)^(-1);
    W = G*Y'*D_L*X;

	% update Dr
	wc = sqrt(sum(W.*W,2)+eps);
	Dr = 0.5./wc;
    % Dr = diag(1./sqrt(sum(W.*W,2)+eps))*0.5;
    
	% update Dl
	% E = X*W-X;
    E = Y*W-X;
	ec = sqrt(sum(E.*E,2)+eps);
	Dl = 0.5./ec;
    % Dl = diag(1./sqrt(sum(E.*E,2)+eps))*0.5;

	obj(iter) = sum(ec) + lambda*sum(wc);
end


end

