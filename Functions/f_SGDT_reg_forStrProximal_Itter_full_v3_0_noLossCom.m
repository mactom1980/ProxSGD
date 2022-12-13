function [U,V,W] = f_SGDT_reg_forStrProximal_Itter_full_v3_0_noLossCom...
    (NNZ_i,NNZ_j,NNZ_k,IT,X,U0,U0_ind,V0,V0_ind,W0,W0_ind,h,g,...
    flg_X_sparse_i,flg_X_sparse_j, flg_X_sparse_k, C, I, J, K)

rand('state',0);

x_vals0 = X.vals;
x_subs = X.subs;
Nzer = length(x_vals0);

U00 = U0; 
V00 = V0; 
W00 = W0;


for it=1:IT %it counts how many SGD will be excecuted during the proximal
%step
    IndexOfNzeros = randperm(Nzer);
    x_vals = x_vals0(IndexOfNzeros');

    for Indnn=1:Nzer %SGD itterations. this is a SGD epoch
        d=x_subs(IndexOfNzeros(Indnn),:);
        ii=d(1);
        jj=d(2);
        kk=d(3);
        
        if flg_X_sparse_i==1
            i = findInSorted(U0_ind,ii);
        else
            i = f_convert_iIndex(ii, C, I);
        end
        if flg_X_sparse_j==1
            j = findInSorted(V0_ind,jj);
        else
            j = f_convert_iIndex(jj, C, J);
        end
        if flg_X_sparse_k==1
            k = findInSorted(W0_ind,kk);
        else
            k = f_convert_iIndex(kk, C, K);
        end
        
        y=x_vals(Indnn);
        S = -2*( y - sum(U0(i,:).*V0(j,:).*W0(k,:)) );
        U_ir = U0(i,:) - h*Nzer*( ( S*( V0(j,:).*W0(k,:) ) ) + (1/g)*( (U0(i,:)-U00(i,:))/NNZ_i(ii) ) );
        V_jr = V0(j,:) - h*Nzer*( ( S*( U0(i,:).*W0(k,:) ) ) + (1/g)*( (V0(j,:)-V00(j,:))/NNZ_j(jj) ) );
        W_kr = W0(k,:) - h*Nzer*( ( S*( U0(i,:).*V0(j,:) ) ) + (1/g)*( (W0(k,:)-W00(k,:))/NNZ_k(kk) ) );
        
        %this is a test for convergance, but has a big computational cost
        %for use only in the parameter tuning phase
        
%         if sum(isnan(U_ir)+isinf(U_ir))>0
%             disp('not converge')
%             return
%         end

       %rows of matrices updates
       U0(i,:) = U_ir;
       V0(j,:) = V_jr;
       W0(k,:) = W_kr;
    end
end

U=U0;
V=V0;
W=W0;

end