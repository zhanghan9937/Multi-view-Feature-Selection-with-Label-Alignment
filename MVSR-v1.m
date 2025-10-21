function [obj,W,id,Z,time1] = MVSR(Xl, YL, Xu, W, option)
%adaptive semi-supervised feature selection code
%Xl: labeled data :d*nl*v
%YL: label matrix, c*nl
%Xu: Unlabeled data, d*nu*v
%opt:  
%      -lambda: the parameter of l_{2,1} norm
%      -eta: the real class number
%      -mu: regualrization parameter 1, 10, 100, 1000, 10000
%      -outMaxIter & inMaxIter # of iterations
%Written by Han Zhang 2024/9/14
n_view = length(Xu);
%% Initialization
alpha = (1/n_view)*ones(n_view,1);
cluster = option.c;
nl = size(Xl{1},2);
nu = size(Xu{1},2);
n = nl+nu;
E = eye(cluster);
for v = 1:n_view
    X{v} = [Xl{v},Xu{v}];
    dim(v) = size(Xl{v},1);
    D{v} = eye(dim(v));
    for i = 1:dim(v)
        D{v}(i,i) = (.5)*(W{v}(i,:)*W{v}(i,:)'+eps).^(-.5);
    end
end 

opt = [];
opt.NeighborMode = 'KNN';
opt.k = 5;
opt.WeightMode = 'HeatKernel';
opt.t = 1;

LL = zeros(n);
for v = 1:n_view
    S{v}=constructW(X{v}', opt);
    S{v} = (S{v}+S{v}')./2;
    Ds{v} = diag(sum(S{v}, 1));
    Ls{v} = Ds{v} - S{v};
    LL = LL+Ls{v};
end
label = kmeans(Xu{1}',cluster);
Zu = zeros(nu,cluster);
for i = 1:nu
    Zu(i,label(i)) = 1;
end
Z = [YL; Zu];
Q = eye(cluster,cluster);
Fu = zeros(nu,cluster);
F = [YL; Fu];
Ss = diag(sum(F,2));
beta = ones(cluster,1);
%% Main Code
obj = zeros(option.outMaxIter, 1);
itr = 1;
%[obj(itr,1)] = compute_obj(alpha, beta, W, D, Ss, X, F, Z, LL, Q, option.lambda, option.eta, option.mu);
%itr = itr +1;
time0=tic;
for iter = 1: option.outMaxIter    
    %% Update beta
    for v = 1:n_view
        pred(:,:,v)=alpha(v)*W{v}'*X{v};
    end
    pred = sum(pred,3);
    a = sum(sum(F));
    b = sum(F,2);
    c = sum(F,1);
    beta = (E*c'-pred*b)./a;
    %% Update F
    for j=1:cluster
        v1=pred'+repmat(beta, 1, n)'-repmat(E(:,j),1,n)';
        G(:,j)=sum(v1.*v1,2);%p(i,k)
    end
    H = Z*Q;
    M = (2*option.mu*H-G)./(2*(option.mu+option.theta));
    Mu = M(nl+1:n,:);
    for i = 1:nu
        [Fu(i,:)] = EProjSimplex_new1(Mu(i,:));
    end
    F = [YL; Fu];
    %% update W
%     ob = zeros(option.inMaxIter,1);
    for it = 1: option.inMaxIter
        for v = 1:n_view   
            Xt{v} = alpha(v)*X{v};
        end
        for v = 1:n_view   
            for i = 1:dim(v)
                 D{v}(i,i)= (.5)*(W{v}(i,:)*W{v}(i,:)'+eps).^(-.5);
            end
            Ss = diag(sum(F,2));  
            A = Xt{v}*Ss*Xt{v}'+option.lambda*D{v};     
            B = 0;
            for r = 1:n_view
                if r ~= v
                    temp = Xt{v}*Ss*Xt{r}'*W{r};                   
                    B = B + temp;
                end
            end
            B = B+2*Xt{v}*ones(n,1)*beta'-2*Xt{v}*F; 
            B = -B./2;
            W{v} = GPI(A,B,W{v});                   
        end
    end
    %% update alpha
    U = cell(1,n);
    for i = 1:n
        for v = 1:n_view
            ui = W{v}'*X{v};
            U{i} = [U{i},ui(:,i)];
        end
    end
    As = 0;
    for i = 1:n
        temp = U{i}'*U{i}*Ss(i,i);
        As = As+temp;          
    end
    for i = 1:n   
        for j = 1:cluster
            temp2(:,i) = U{i}'*F(i,j)*(E(:,j)-beta);
        end
    end
    Bs = 2*sum(temp2, 2);
    [alpha(:,iter+1), ~] =SimplexQP_acc(As, Bs);
    %% update Z
    A = Q*Q';
    B = Q*F(nl+1:end,:)';
    Lu = LL(nl+1:end, nl+1:end);
    param = option.mu/option.eta;
    [Zu] = CD(Lu, A, B, param, Zu);
    Z = [YL; Zu];
    %% update Q
    [Q, ~] = ALM_NMF(F, Z, Q);
    [obj(itr,1),~,~,~,~,~] = compute_obj(alpha, beta, W, D, Ss, X, F, Z, LL, Q, option.lambda, option.eta, option.mu,option.theta);
    itr = itr +1;
end
time1=toc(time0);
Wt = [];
for v = 1:n_view
    Wt = [Wt; W{v}];
end
sumWt = sum(Wt.^2,2);
[~,id] = sort(sumWt,'descend');
%plot(obj);
end

