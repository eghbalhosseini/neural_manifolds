%% Compute simulation capacity of the general manifold.
%% Author: SueYeon Chung. Jul 10, 2018

function [output] = manifold_simcap_analysis(XtotT, options)
    % XtotT: cell file. Each cell is M_i by N matrix, where M_i is number of
    % samples in i_{th} manifold, and N is the feature dimension. 
    n_rep=options.n_rep;
    seed0=options.seed0; 
    %%
    flag_NbyM = options.flag_NbyM; 
    P=length(XtotT); 
    for ii=1:P
        if ~flag_NbyM 
            Xtot{ii}=XtotT{ii}';    
        else
            Xtot{ii}=XtotT{ii}; 
        end
        N=size(Xtot{ii},1); 
    end
    clear XtotT; 
    %%
    Xori=[]; 
    for pp=1:P
        M_vec(pp)=size(Xtot{pp},2); %% #(samples) vec. 
        Xori=[Xori Xtot{pp}];  %% All data.
    end
    M0=sum(M_vec); % Total number of samples. 
    X0=mean(Xori,2); %Global Mean 
    clear Xori; 
    centers=nan(N,P);
    for pp=1:P
        Xtot0{pp}=Xtot{pp}-repmat(X0, [1 M_vec(pp)]);   % Previously X 
        centers(:,pp)=mean(Xtot0{pp},2); 
    end
    clear Xtot;
    %% 
    Nmin=2; Nmax=N; p_tol=0.05; flag_n=2; 
    
    [Nc, N_vec, p_vec]= bisection_Nc_general(Xtot0, n_rep, Nmin, Nmax, p_tol, seed0, flag_n); 
    clear Xtot0; 
    
    asim=P/Nc; 
    %    save('checkfun.mat')
    %     Nc0=interp1(p_vec(1,[end-1,end]), N_vec(1,[end-1,end]), 0.5); 
    ind_mid=[max(find(p_vec<0.5)) min(find(p_vec>=0.5))];
    
    if isempty(ind_mid) | isnan(p_vec)
        Nc0=nan; 
        asim0=nan; 
    else
        Nc0=interp1(p_vec(ind_mid), N_vec(ind_mid), 0.5); 
        asim0=P/Nc0; 
    end
    fprintf('a_sim=%2.2f, asim0(interp)=%2.2f% \n',asim, asim0)
    
    output.asim0=asim0;
    output.P=P;
    output.Nc0=Nc0; 
end

function [Ncur, Nall_vec, pall_vec ]= ...
    bisection_Nc_general(Xtot, n_rep, Nmin, Nmax, p_tol, seed0, flag_n)
P=length(Xtot); 
%f_pdiff = @(Nr) (compute_sep_Nc(X, X_s, P, Nr, n_rep)-0.5); 
%[p_conv] = compute_sep_Nc(X, X_s, P, N_cur, n_rep)
f_pdiff = @(Nr) (compute_sep_Nc_general(Xtot, Nr, n_rep, seed0, flag_n)-0.5); 

% provide the equation you want to solve with R.H.S = 0 form. 
% Write the L.H.S by using inline function
% Give initial guesses.
% Solves it by method of bisection.
% A very simple code. But may come handy

fmin=f_pdiff(Nmin);
fmax=f_pdiff(Nmax); 
pmin_vec(1)=fmin+.5;
pmax_vec(1)=fmax+.5; 
Nmin_vec(1)=Nmin;
Nmax_vec(1)=Nmax;
if pmax_vec(1)==0
    fprintf('Maximum N gives 0 separability. Need more neurons. \n');
    Ncur=nan;
    Nall_vec=nan;
    pall_vec=nan; 
    return;
end

if fmin*fmax>0 
    disp('Wrong choice of Nmin and Nmax.\n')
    Ncur=nan; fcur=nan; Ncur_vec=nan; pcur_vec=nan;
else
    Ncur = round((Nmin + Nmax)/2);
    fcur = f_pdiff(Ncur); 
    err = abs(fcur);
    Ncur_vec=Ncur; pcur_vec=fcur+.5;
    kk=1; 
    dN=1000; 
    %save('checkBisection.mat')
    while (err > p_tol) && (dN > 1) && kk<100 
        kk=kk+1; 
       
        fprintf('%dth bisection run, P=%d, Nmin=%d, pmin=%.2e, Nmax=%d, pmax=%.2e.\n', ...
           kk, P, Nmin, fmin+0.5, Nmax,  fmax+0.5)
%        if f_pdiff(Nmin)*f_pdiff(Ncur)<0 
       if fmin*fcur<0
           Nmax = Ncur; fmax=fcur; 
       else
           Nmin = Ncur; fmin=fcur;       
       end
       pmin_vec(kk)=fmin+.5;
        pmax_vec(kk)=fmax+.5; 
        Nmin_vec(kk)=Nmin;
        Nmax_vec(kk)=Nmax;
       
        Ncur = round((Nmin + Nmax)/2); 
        fcur = f_pdiff(Ncur);
%         err = abs(f_pdiff(Ncur));
        
        err=abs(fcur); 
        fprintf('                 err=%2.2f, p_tol=%2.2f.\n', ...
           err,p_tol)
        dN = (Nmax-Nmin); 
        err_vec(kk)=err;
        Ncur_vec(kk)=Ncur; 
        pcur_vec(kk)=fcur+.5; 
       
    end
end

Nall_vec0 = [Ncur_vec Nmin_vec Nmax_vec];
pall_vec0 = [pcur_vec pmin_vec pmax_vec];

[Nall_vec iuniq] = unique([Ncur_vec Nmin_vec Nmax_vec]);
pall_vec = pall_vec0(iuniq); 

end

function [p_conv] = compute_sep_Nc_general(Xtot, N_cur, n_rep, seed, flag_n)
rng(seed)
% Xtot: 1xP cell, Xtot{1}=[N M]
P=length(Xtot); 
% N=length(Xtot{1}); 
N=size(Xtot{1},1); 
%% facilitate.!
if N_cur>1500
    n_rep=5; 
end
%%
for ll=1:n_rep
    indpAll{ll}=randperm(P,P/2);
end
%%
Xtemp=[];
for pp=1:P
    Xtemp=[Xtemp Xtot{pp}];
end

%%
for ll=1:n_rep
        indp=indpAll{ll};
        labels00=-ones(1,P); 
        labels00(1,indp)=1; 
        
        
        if flag_n==1
%             ind_a=find(mean(Xtemp.^2,2)>1e-3); % active neurons to these Psub objects
%             ind_a=find(std(Xtemp.^2,[],2)>1e-4);
            ind_a=find(std(Xtemp,[],2)>mean(std(Xtemp,[],2))*1e-4);
        end
        
        if flag_n==2
            %W=orth(randn(N,N_cur));
            W=randn(N,N_cur);
            W=W./repmat(sqrt(sum(W.^2,1)),[N 1]);
        elseif flag_n==3
            [U,S,V] = svd(Xtemp,0);
            Nsvd=size(U,2); 
            if Nsvd<N_cur % Unlikely
                U=[U orth(randn(N, N_cur-Nsvd))];
            end
            W=U(:,1:N_cur); 
        elseif flag_n==1
            Na=length(ind_a); % Number of 'active' neurons for this P 
            fprintf('N=%d, Nactive=%d, N_cur=%d\n',N, Na, N_cur)
            if Na>=N_cur
                indn=ind_a(randperm(Na,N_cur)); % subsample among these active neurons
            else
                fprintf('---> Not enough active neurons, include inactive neurons.\n')
                indn=(randperm(N,N_cur)); % subsample among these active neurons
            end
        end
        %save('check_compute_Nc.mat','W','Xtot')
        for ii=1:P
              if flag_n==1
                Xsub{ii}=Xtot{ii}(indn,:);
              elseif (flag_n==2) | (flag_n==3)
                Xsub{ii}=W'*Xtot{ii};  %clear W; 
              end
        end
        %clear Xtot; 
        fprintf([flag_n sprintf(', N_cur=%d, %dth run: ', N_cur, ll)]); 
        [sep0, w0, bias0,margin0] = check_data_separability_general(Xsub, labels00);
        sep_vec(ll)=sep0; 
        marg_vec(ll)=margin0;
        fprintf(' cum_sep=%2.2e\n',mean(sep_vec(1:ll)));
end
p_conv=mean(sep_vec); 
end

function [sep, w, bias, margin] = ...
    check_data_separability_general(Xsub, labels0)
pp=find(labels0==1); nn=find(labels0==-1);
gpp=sprintf(' %d', pp);gnn=sprintf(' %d', nn);
%ind_r=find(labels0~=0);
%labels=labels0(ind_r);
%X_NPM=X_NPM0(:,ind_r,:);  
P=length(Xsub); 
N=size(Xsub{1},1); 
%N=size(X_NPM,1); M=size(X_NPM,3); P=size(X_NPM,2); 
X=[];y=[];
for ii=1:P
    M=size(Xsub{ii},2); 
    X=[X Xsub{ii}];
    y=[y labels0(ii)*ones(1,M)]; 
    M_vec(ii)=M;
end
% X = reshape(X_NPM,[N P*M]);
% y = repmat(labels, [1 M]); 

w_ini=zeros(N,1);
bias_ini=0;
kappa=0;
C0=100; 
xi_ini=zeros(M*P,1);
tolerance=1e-8; 
%save('check_data_sep.mat')
try
%     [sep, sep_slack, w, bias, xi, margin, flag, minima] ...
%     =find_svm_cplexqp_sep_primal_slackb(X, y, w_ini, bias_ini, kappa, C0, xi_ini, tolerance); 
    
    [sep, w, margin, flag, u, bias] = ...
    find_svm_cplexqp_sep_primal_wb(X, y, w_ini, kappa, tolerance, bias_ini, 1); 

catch error 
    flag_wb=1; m_ini=1; m_add=5; tmax=500; 
    [sep, w, margin, flag_out, u_out, bias] ...
     = compute_m4(X, y, w_ini, kappa, tolerance, bias_ini, flag_wb, m_ini, m_add, tmax); 
    %return; 
end

fprintf('Ntot=%d, P=%d, separable? =%d, marg=%2.2e',...
    N, P, sep, margin)
% if N>20
%     X(1:5,:)
% end
% if N>150
%     fprintf('paused.')
%     pause
% end
end

function  [separable, w, margin, flag, u, b] ...
 = compute_m4(X, y, w_ini, kappa, tolerance, b_ini, flag_wb, m_ini, m_add, tmax)
% function [sep, sep_slack, w, bias, xi, margin, flag, minima] ...
%     =find_svm_cplexqp_sep_primal_slackb(X, y, w_ini, bias_ini, kappa, C0, xi, tolerance)

[N PP]=size(X);
PP1=find(y==1); m1=length(PP1);
PP2=find(y==-1); m2=length(PP2);
m1_ini=min(m1, round(m_ini/2)); 
m2_ini=min(m2, round(m_ini/2)); 
%save('checkm4.mat')
iini=[PP1(randperm(m1, m1_ini)) PP2(randperm(m2, m2_ini))];

%iini=randperm(PP, m_ini);
Xini=X(:, iini); 
yini=y(:, iini); 
iall=iini; 
itot=1:PP; 
marg_tol=1e-8; 
X0 = Xini; y0= yini; 
tt=0; margin_old=0; 
% save('checkm4.mat')
% pause
while tt<= tmax     
 tt=tt+1; 
 [separable, w, margin, flag, u, b] = ...
    find_svm_cplexqp_sep_primal_wb(X0, y0, w_ini, kappa, tolerance, b_ini, flag_wb);

 if tt==tmax
     fprintf('t_max reached.\n')
     separable=(rand(1)>0.5);
 end

 if separable
     irest = setdiff(itot, iall); 
     hrest = ((w'*X(:, irest)+b).*y(:,irest))/ norm(w); 
     if all(hrest > margin)
         fprintf('all fields larger than margin.\n')
         break
     elseif norm(margin - margin_old) < marg_tol 
         fprintf('margin converged. \n')
         break 
     elseif sum(hrest < margin - marg_tol)>m_add 
         [hs, is] = sort(hrest, 'ascend');
         iadd=irest(is(1:m_add)); 
     else
         [hw, iw]=find(hrest<margin - marg_tol); 
         iadd=irest(iw); 
     end 
     
     X0 = [X0 X(:,iadd)];
     y0 = [y0 y(:,iadd)];
     margin_old = margin; 
     %fprintf('Nadded=%d, margin=%.3f\n', length(iadd), margin); 
 else 
     fprintf('Not separable.\n')
     break;
 end 
end


end

function [seperable, w, margin, flag, u, b] = ...
    find_svm_cplexqp_sep_primal_wb(X, y, w_ini, kappa, tolerance, b_ini, flag_wb)

%addpath('/home/phys/users/sueyeon.chung/cplexdir1263/cplex/matlab/x86-64_linux/')
    if nargin < 4
        kappa = 0;
    end
    if nargin < 5
        tolerance = 1e-8;  
    end
    
    minimal_tolerance = tolerance;     
    M = size(X, 2);   % P+1
    N = size(X, 1);
    
    assert(all(size(X) == [N, M]), 'x must be NxM');
    
    assert(all(size(y) == [1, M]), 'y must be M labels');
    assert(all(abs(y) == 1), 'y must be +1/-1');
    
    %% Keep 
    
    options = cplexoptimset('cplex');
    %options.display = 'on';
    options.MaxIter = 1e25;
    options.qpmethod = 4; 
    %options.MaxTime = 200; 
    
    scale = 1;
    %  The feasibility tolerance specifies the amount by which a solution can violate its 
    % bounds and still be considered feasible. The optimality tolerance specifies the amount 
    % by which a reduced cost in a minimization problem can be negative.
    if tolerance < minimal_tolerance
        scale = minimal_tolerance / tolerance;
        options.simplex.tolerances.feasibility = minimal_tolerance;
        options.simplex.tolerances.optimality = minimal_tolerance;
    else
        options.simplex.tolerances.feasibility = tolerance;
        options.simplex.tolerances.optimality = tolerance;
    end

    
    
      try
           
%         [u,minima,flag,output] = cplexqp(0.5*scale*H0,scale*f, -scale*A, -scale*b, [], [], scale*lb, [], [], options);
     %% minimize wHw+ fw s.t. Aineq*w<=bineq, Aeq*w= beq, lb <= w <= ub
     %% WT: min |w|^2 while -(yx-k0)w<=-1 (-yxw+k<=-1)   (yxw>=1+k) (yxw-k>=1)
     %% x = cplexqp(H,f,Aineq,bineq,Aeq,beq,lb,ub,x0,options)
        
     
    Hb = eye(N+1, N+1); Hb(N+1,N+1)=0; 
    fb = zeros(1,N+1);    
    
    if flag_wb ==1 
        Ab= - diag(y')*[X' ones(M,1)]; 
    else 
        Ab= - diag(y')*[X' zeros(M,1)]; 
    end 
    
    bb = - ones(M, 1);
%     save('checksvm.mat')

%     [ub,minima,flag,output] = cplexqp(0.5*scale*Hb, scale*fb, scale*Ab, scale*bb, [], [], [], [], [], options);
    [ub,minima,flag,output] = quadprog(0.5*scale*Hb, scale*fb, scale*Ab, scale*bb, [], [], [], [], [], options);
        if flag_wb ==0 
            ub(end,1)=0; 
        end

         u = ub(1:N,1);
         b_out = ub(end,1); 
        
%           [u,minima,flag,output,lagrangians] = cplexlp(scale*f, -scale*A, -scale*b, [], [], scale*lb, [], [], options);
      catch error        
%        
        ub = zeros(N+1,1); u = ub(1:N,1); b = ub(end,1); 
        w = zeros(N, 1); 
        seperable = 0; 
        margin = NaN;
        flag = -100;
        
        if strfind(error.message, '1256: Basis singular.')
            fprintf('Warning: Basis singular\n');
        else
            warning('cplex failed: %s', error.message);
        end
        return;
    end        

    minima = minima / scale;
    if flag < 0
        w = zeros(N, 1); b =0; 
        seperable = false;
        if nargout > 2
            margin = NaN;
        end
        return;
    end
    
    assert(all(size(u) == [N, 1]));
    % Get the results from the output
    wb=ub; w=wb(1:N,1); b=wb(N+1,1); 
%    w = u(1:N)-u((N+1):2*N);
    %figure; plot(u(1:2*N)); xlabel('u(1:2*N)')
    %w = u(1:N); 
    %w = u(N+1:2*N); 
% If there was no timeout, the optimal weights should well-converge
%     if flag == 1
%         r = u - lb;
%         assert(all(r >= -1e-5), sprintf('Lower bound violated: %1.1e (%d)', min(r(:)), flag));
%     end
    if nargout > 2
%         margin = min((w'*X - b_out).*y) / norm(w, 2);
        b = b_out/ norm(w,2); w = w / norm(w,2); 
        margin = min(((w'*X+b)./norm(w')).*y ) ;
    end    
    seperable = all(sign((w'*X+b)./norm(w'))==y); 
    
    %seperable = all(sign(w(1:end-1)'*Xb(1:N-1,:))==y);    
end