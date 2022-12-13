function [s_loss] = f_compute_real_Loss_StrSGD_v3(U0,U0_ind,V0,V0_ind,W0,W0_ind,X,...
    flg_X_sparse_i,flg_X_sparse_j, flg_X_sparse_k, C, I, J, K)
    
    x_subs = X.subs;
    x_vals0 = X.vals;
    Nzer = length(x_vals0);
    s_loss=0;

    for Indnn=1:Nzer
        d=x_subs(Indnn,:);
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
        
        y=x_vals0(Indnn);
        loss_Xijk = ( y -sum( U0(i,:).*V0(j,:).*W0(k,:) ))^2;
        s_loss = s_loss + loss_Xijk;
        
    end

end