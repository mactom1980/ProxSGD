function [U,V,W] = f_SGDT_reg_forProximal_Itter_v1_0_noLossCom...
    (NNZ_i,NNZ_j,NNZ_k,IT,x_vals0,x_subs,Nzer,U0,V0,W0,h,g)

rand('state',0);

U00 = U0; 
V00 = V0; 
W00 = W0;

for it=1:IT %it counts how many SGD will be excecuted during the proximal
%step
    IndexOfNzeros = randperm(Nzer);
    x_vals = x_vals0(IndexOfNzeros');

    for Indnn=1:Nzer %SGD itterations. this is a SGD epoch
        d=x_subs(IndexOfNzeros(Indnn),:);
        i=d(1);
        j=d(2);
        k=d(3);
        
        y=x_vals(Indnn);
        S = -2*( y - sum(U0(i,:).*V0(j,:).*W0(k,:)) );
        U_ir = U0(i,:) - h*Nzer*( ( S*( V0(j,:).*W0(k,:) ) ) + (1/g)*( (U0(i,:)-U00(i,:))/NNZ_i(i) ) );
        V_jr = V0(j,:) - h*Nzer*( ( S*( U0(i,:).*W0(k,:) ) ) + (1/g)*( (V0(j,:)-V00(j,:))/NNZ_j(j) ) );
        W_kr = W0(k,:) - h*Nzer*( ( S*( U0(i,:).*V0(j,:) ) ) + (1/g)*( (W0(k,:)-W00(k,:))/NNZ_k(k) ) );
        
        %this is a test for convergance, but has a big computational cost
        %for use only in the parameter tuning phase
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