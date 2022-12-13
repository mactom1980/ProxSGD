function [U,V,W] = f_SGDT_v3_0...
    (X,U0,U0_ind,V0,V0_ind,W0,W0_ind,h,...
    flg_X_sparse_i,flg_X_sparse_j, flg_X_sparse_k, C, I, J, K)

%Stochastic Gradient Descent Method for Tensors
%
%Input Parameters
%
%IT = number of epochs
%
%X = the sparse Tensor
%Nzer = Number of nonZero elements
%U0, V0, W0 = initial values for the product factors
%h = stepp of gradient descent method
%X_start = the initial image Tensor before the missing values
%
%output values
%
%Loss_real = The real realtive loss between the training tensor (i.e. the
%tensor who's missing values are ignored and the computed tensor who's
%values that correspond to the missing values are ignored
%Loss_tens = The relative loss between the full initial tensor and the full
%computed tensor
%X_ten = the full computed tensor



% Loss_ten = zeros(IT,1);
% Loss_per_IT_Tensor_real = zeros(IT,1);


rand('state',0);
x_subs = X.subs;
x_vals0 = X.vals;

Nzer = length(x_vals0);

IndexOfNzeros = randperm(Nzer);
x_vals = x_vals0(IndexOfNzeros');
% tic
for Indnn=1:Nzer

    d=x_subs(IndexOfNzeros(Indnn),:);
    i=d(1);
    j=d(2);
    k=d(3);
    
    if flg_X_sparse_i==1
        i = findInSorted(U0_ind,i);
    else
        i = f_convert_iIndex(i, C, I);
    end
    if flg_X_sparse_j==1
        j = findInSorted(V0_ind,j);
    else
        j = f_convert_iIndex(j, C, J);
    end
    if flg_X_sparse_k==1
        k = findInSorted(W0_ind,k);
    else
        k = f_convert_iIndex(k, C, K);
    end

    y=x_vals(Indnn);
    S = -2*( y - sum(U0(i,:).*V0(j,:).*W0(k,:)) );
    
    U_ir = U0(i,:) - h*Nzer*( S*( V0(j,:).*W0(k,:) ) );
    V_jr = V0(j,:) - h*Nzer*( S*( U0(i,:).*W0(k,:) ) );
    W_kr = W0(k,:) - h*Nzer*( S*( U0(i,:).*V0(j,:) ) );
    
   %rows of matrices updates
   U0(i,:) = U_ir;
   V0(j,:) = V_jr;
   W0(k,:) = W_kr;
   
   
end

U=U0;
V=V0;
W=W0;

end