function [T, it, EpochTime, Loss_real] = StrSGD_v5...
    (C, flg_Img, Data_file, flg_rand_state, R, q, IT, z_perc, precision)

% This function is implementing the Stratified Stochastic Gradient
% Descent Method from the paper:
% 
% 
% T. Papastergiou and V. Megalooikonomou, "A distributed proximal gradient descent method for tensor completion", 
% 2017 IEEE International Conference on Big Data (Big Data), Boston, MA, 2017, pp. 2056-2065, doi: 10.1109/BigData.2017.8258152
% 
%
% The training set is partitioned in d^2 independent strata
% and in each stratum the Proximal Operator is calculated in parallel.

% Arguments of the function
% C: Nr of workers
% flg_Img: if 1 then the data file is a image, if 2 the file is a hyperspectral image
%       else the data file is a .mat file, that must contain a tensor named X
% Data_file: the full path to the file.
% flg_rand_state: if flg_rand_state>=0  rand state is put to flg_rand_state,
%       if flg_rand_state == -1 then no rand state is used
% R: the rank of the PARAFAC Decomposition
% q: the exponent in the learning rate. the learning rate is calculated as
%       follows: 1/(2^q)
% IT: the number of maximum iterations of the method
% z_perc: the percentage of the missing values (applicable only in images
%       and hyperspectral images
% precision: termination criterion of the algorithm

if flg_rand_state>=0
    rand('state',flg_rand_state);
end

disp('loading and preprocessing data');

if flg_Img == 1
    [I, J, K, X,~] = f_imgReadAndPreprossecing(Data_file,z_perc);
elseif flg_Img == 2
    [X, IndexOfZeros, I, J, K] = f_specImgPrepross(Data_file, 1, 1280, 1, 307, z_perc);
else
    load(Data_file);
    f = X.size;
    I = f(1);
    J = f(2);
    K = f(3);
end

execution_time = tic;

h = 1/(2^q);

disp('one time jobs:');

flg_X_sparse_i = 1;
flg_X_sparse_j = 1;
flg_X_sparse_k = 1;

X_un_subs_i = unique(X.subs(:,1));
X_un_subs_j = unique(X.subs(:,2));
X_un_subs_k = unique(X.subs(:,3));

if length(X_un_subs_i) == I %>=.9*I
    flg_X_sparse_i = 0;
    disp('Dimension I is not sparse');
    clear X_un_subs_i;
end
if length(X_un_subs_j) == J %>=.9*J
    flg_X_sparse_j = 0;
    disp('Dimension J is not sparse');
    clear X_un_subs_j;
end
if length(X_un_subs_k) == K %>=.9*K
    flg_X_sparse_k = 0;
    disp('Dimension K is not sparse');
    clear X_un_subs_k;
end

%partition of U0,V0,W0 in to C sequential blocks.

if flg_X_sparse_i == 0
    U0_part = cell(1,C);
else
    U0_part = cell(2,C);
end
if flg_X_sparse_j == 0
    V0_part = cell(1,C);
else
    V0_part = cell(2,C);
end
if flg_X_sparse_k == 0
    W0_part = cell(1,C);
else
    W0_part = cell(2,C);
end

index_part_X = floor(I/C);
index_part_Y = floor(J/C);
index_part_K = floor(K/C);

% This function is implementing the Stratified Proximal Stochastic Gradient
% Descent Method from the paper:
% 
% 
% T. Papastergiou and V. Megalooikonomou, "A distributed proximal gradient descent method for tensor completion", 
% 2017 IEEE International Conference on Big Data (Big Data), Boston, MA, 2017, pp. 2056-2065, doi: 10.1109/BigData.2017.8258152
% 

% The training set is partitioned in d^2 independent strata
% and in each stratum the Proximal Operator is calculated in parallel.

% Arguments of the function
% C: Nr of workers
% flg_Img: if 1 then the data file is a image, if 2 the file is a hyperspectral image
%       else the data file is a .mat file, that must contain a tensor named X
% Data_file: the full path to the file.
% flg_rand_state: if flg_rand_state>=0  rand state is put to flg_rand_state,
%       if flg_rand_state == -1 then no rand state is used
% R: the rank of the PARAFAC Decomposition
% q: the exponent in the learning rate. the learning rate is calculated as
%       follows: 1/(2^q)
% IT: the number of maximum iterations of the method
% g: the proximal coefficient, see the paper
% ITperPStep: SGD Iterations per proximal step, see the paper
% z_perc: the percentage of the missing values (applicable only in images
%       and hyperspectral images
% precision: termination criterion of the algorithm

%netflix U0 initialization
U0 = randi([1 5], I,R);

for i=0:C-1
    if i==C-1
        flg_mod_X = mod(I,C);
%         flg_mod_Y = mod(J,C);
%         flg_mod_K = mod(K,C);
    else
        flg_mod_X = 0;
%         flg_mod_Y = 0;
%         flg_mod_K = 0;
    end
    
%     U0_part{i+1} = zeros(index_part_X+flg_mod_X,R);
%     U0_part{i+1} = U0(i*index_part_X+1:(i+1)*index_part_X + flg_mod_X,:);
    if flg_X_sparse_i==0
        U0_part{i+1} = zeros(index_part_X+flg_mod_X,R);
        U0_part{i+1} = U0(i*index_part_X+1:(i+1)*index_part_X + flg_mod_X,:);
%         clear U0;
    else
        [a,b] = findInSorted(X_un_subs_i, [i*index_part_X+1, (i+1)*index_part_X+flg_mod_X]);
        U0_part{1,i+1} = zeros(b-a+1,R);
        U0_part{1,i+1} = U0(X_un_subs_i(a:b),:);
        U0_part{2,i+1} = X_un_subs_i(a:b);
%         clear U0;
    end
    
end
clear U0;
clear X_un_subs_i;


general V0 initialization
V0 = rand(J,R);

%netflix V0 initialization
% V0 = randi([1 5], J,R);

for i=0:C-1
    if i==C-1
        flg_mod_Y = mod(J,C);
    else
        flg_mod_Y = 0;
    end
    
    if flg_X_sparse_j == 0
        V0_part{i+1} = zeros(index_part_Y+flg_mod_Y,R);
        V0_part{i+1} = V0(i*index_part_Y+1:(i+1)*index_part_Y+flg_mod_Y,:);
    else
        [a,b] = findInSorted(X_un_subs_j, [i*index_part_Y+1, (i+1)*index_part_Y+flg_mod_Y]);
        V0_part{1,i+1} = zeros(b-a+1,R);
        V0_part{1,i+1} = V0(X_un_subs_j(a:b),:);
        V0_part{2,i+1} =  X_un_subs_j(a:b); 
    end
    
end
clear V0;
clear X_un_subs_j;

W0 = rand(K,R);

for i=0:C-1
    if i==C-1
        flg_mod_K = mod(K,C);
    else
        flg_mod_K = 0;
    end
    
    if flg_X_sparse_k == 0
        W0_part{i+1} = zeros(index_part_K+flg_mod_K,R);
        W0_part{i+1} = W0(i*index_part_K+1:(i+1)*index_part_K + flg_mod_K,:);
    else
        [a,b] = findInSorted(X_un_subs_k , [i*index_part_K+1, (i+1)*index_part_K+flg_mod_K]);
        W0_part{1,i+1} = zeros(b-a+1,R);
        W0_part{1,i+1} = W0(X_un_subs_k(a:b),:);
        W0_part{2,i+1} = X_un_subs_k(a:b);
    end
end

clear W0;
clear X_un_subs_k;

%partitioning the tensor in strata
X_sp_part = cell(C,C^2);

for ip=0:C-1
    for s=0:C^2-1
        X_sp_part{ip+1,s+1} = sptensor([I J K]);
        [X_sp_part{ip+1,s+1}] = f_find_a_stratum(s,ip,I,J,K,C,X);
    end
end

norm_X = norm(X);
clear X;

Loss_real = zeros(1,IT);
EpochTime = zeros(1,IT);

%Composite Variables
U0_part_C = Composite(C);
V0_part_C = Composite(C);
W0_part_C = Composite(C);

U0_part_ind_C = Composite(C);
V0_part_ind_C = Composite(C);
W0_part_ind_C = Composite(C);

h_C = Composite(C);

X_sp_part_C = Composite(C);
s_C = Composite(C);
it_C = Composite(C);

I_C = Composite(C);
J_C = Composite(C);
K_C = Composite(C);
C_C = Composite(C);

flg_X_sparse_i_C = Composite(C);
flg_X_sparse_j_C = Composite(C);
flg_X_sparse_k_C = Composite(C);

%send U0_part to the workers. send also h to the workers.
for i=1:C
    if flg_X_sparse_i==1
        U0_part_C{i} = U0_part{1,i};
        U0_part_ind_C{i} = U0_part{2,i};
    else
        U0_part_C{i} = U0_part{i};
        U0_part_ind_C{i} = [];
    end
    if flg_X_sparse_j==0
        V0_part_ind_C{i} = [];
    end
    if flg_X_sparse_k==0
        W0_part_ind_C{i} = [];
    end
    h_C{i} = h;
    X_sp_part_C{i} = X_sp_part(i,:);
    
    I_C{i} = I;
    J_C{i} = J;
    K_C{i} = K;
    C_C{i} = C;
    
    flg_X_sparse_i_C{i} = flg_X_sparse_i;
    flg_X_sparse_j_C{i} = flg_X_sparse_j;
    flg_X_sparse_k_C{i} = flg_X_sparse_k;
end

clear X_sp_part;
clear U0_part;

for it=1:IT
    disp('epoch:')
    disp(it)
    %here begins an epoch
    for s = 0:(C^2)-1
    %here begins a subepoch
    
        for ip=0:C-1
            %send the appropriate blocks of V0_part and W0_part to the workers. The
            %U0_part blocks have been allready sent to the workers since each
            %worker needs only a unique part of U0
            if flg_X_sparse_j==1
                V0_part_C{ip+1} = V0_part{1,mod(ip+s,C)+1};
                V0_part_ind_C{ip+1} = V0_part{2,mod(ip+s,C)+1};
            else
                V0_part_C{ip+1} = V0_part{mod(ip+s,C)+1};
            end
            
            if flg_X_sparse_k==1
                W0_part_C{ip+1} = W0_part{1,mod(ip+floor(s/C),C)+1};
                W0_part_ind_C{ip+1} = W0_part{2,mod(ip+floor(s/C),C)+1};
            else
                W0_part_C{ip+1} = W0_part{mod(ip+floor(s/C),C)+1};
            end
            %send also the stratum s to the workers
            s_C{ip+1} = s;%this s is the stratum starting from 0 to C^2-1
            it_C{ip+1} = it;
        end


        spmd(C)
            if (it_C==1) && (s_C==0)
                U0_internal = U0_part_C;
                U0_internal_ind = U0_part_ind_C;
                disp('read U0_C')
            end
            %run SGD on the Stratum Selected
            [U0_internal,V0_internal,W0_internal] = f_SGDT_v3_0(X_sp_part_C{s_C+1},...
                U0_internal,U0_internal_ind,V0_part_C,V0_part_ind_C,W0_part_C,W0_part_ind_C,...
                h_C, flg_X_sparse_i_C,flg_X_sparse_j_C, flg_X_sparse_k_C, C_C,I_C, J_C, K_C);
        end
%     collect and concatenate the updates
        for ip=0:C-1
            if flg_X_sparse_j==1
                V0_part{1,mod(ip+s,C)+1} = V0_internal{ip+1};
            else
                V0_part{mod(ip+s,C)+1} = V0_internal{ip+1};
            end
            
            if flg_X_sparse_k==1
                W0_part{1,mod(ip+floor(s/C),C)+1} = W0_internal{ip+1};
            else
                W0_part{mod(ip+floor(s/C),C)+1} = W0_internal{ip+1};
            end
        end
    %here ends a subepoch
    end

%new parallel loss computing using strata
    for s = 0:(C^2)-1
    %here begins a subepoch for loss computing

        for ip=0:C-1
            %send the appropriate blocks of V0_part and W0_part to the workers. The
            %U0_part blocks have been allready sent to the workers since each
            %worker needs only a unique part of U0
            if flg_X_sparse_j==1
                V0_part_C{ip+1} = V0_part{1,mod(ip+s,C)+1};
                V0_part_ind_C{ip+1} = V0_part{2,mod(ip+s,C)+1};
            else
                V0_part_C{ip+1} = V0_part{mod(ip+s,C)+1};
            end
            
            if flg_X_sparse_k==1
                W0_part_C{ip+1} = W0_part{1,mod(ip+floor(s/C),C)+1};
                W0_part_ind_C{ip+1} = W0_part{2,mod(ip+floor(s/C),C)+1};
            else
                W0_part_C{ip+1} = W0_part{mod(ip+floor(s/C),C)+1};
            end
            %send also the stratum s to the workers
            s_C{ip+1} = s;%this s is the stratum starting from 0 to C^2-1
        end

        spmd(C)
             Loss_real_C =  f_compute_real_Loss_StrSGD_v3...
                (U0_internal,U0_internal_ind,V0_part_C,V0_part_ind_C,W0_part_C,W0_part_ind_C,...
            X_sp_part_C{s_C+1},flg_X_sparse_i_C,flg_X_sparse_j_C, flg_X_sparse_k_C,...
            C_C, I_C, J_C, K_C);
         end

        if (s==0)
            Loss_real_tmp = 0;
        end
        
        for ip=1:C
            Loss_real_tmp = Loss_real_tmp + Loss_real_C{ip};
        end
    %here ends a subepoch for loss computatuion
   end
    Loss_real(it) = sqrt(Loss_real_tmp)/norm_X;
%new parallel loss computing using strata ends here
    disp(Loss_real(it));
    if (Loss_real(it)<=precision)
        EpochTime(it) = toc(execution_time);
        break;
    end
    EpochTime(it) = toc(execution_time);
%here ends an epoch

end

if flg_Img==1 || flg_Img==2
    U0_new = zeros(I,R);
    V0_new = zeros(J,R);
    W0_new = zeros(K,R);
    for i=0:C-1
        if i==C-1
            flg_mod_X = mod(I,C);
            flg_mod_Y = mod(J,C);
            flg_mod_K = mod(K,C);
        else
            flg_mod_X = 0;
            flg_mod_Y = 0;
            flg_mod_K = 0;
        end
        U0_new(i*index_part_X+1:(i+1)*index_part_X + flg_mod_X,:) = U0_internal{i+1};
        V0_new(i*index_part_Y+1:(i+1)*index_part_Y + flg_mod_Y,:) = V0_part{i+1};
        W0_new(i*index_part_K+1:(i+1)*index_part_K + flg_mod_K,:) = W0_part{i+1};
    end

end

T = toc(execution_time);

figure('Name','Loss Plot','NumberTitle','off');

title(['StrSGD r=',num2str(R),'  h=', num2str(h),...
    '  C=', num2str(C), ' time=',num2str(T)]);
plot(Loss_real,'r');   %computed with tensors

%ploting the loss versus the wallclock time elapsed

figure('Name','Loss vs Time','NumberTitle','off');

title(['StrPrSGD r=',num2str(R),'  h=', num2str(h),'  C=', num2str(C),...
    'Time=',num2str(T)]);
plot(EpochTime,Loss_real,'r'); 

%output image

if flg_Img == 1
    f_imgDisplay(U0_new, V0_new,W0_new);
end

if flg_Img == 2
    f_spec_ImgDisp(U0_new, V0_new, W0_new, 63, 52, 36);
end

disp(T);

end
