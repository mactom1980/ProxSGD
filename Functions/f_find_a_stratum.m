function [X_sp_part] = f_find_a_stratum(s,ip,I,J,K,C,X)

index_part_X = floor(I/C);
index_part_Y = floor(J/C);
index_part_K = floor(K/C);
    
if ip==C-1
    mod_flg_I = mod(I,C);
else
    mod_flg_I = 0;
end
if mod(ip+s,C)==C-1
    mod_flg_J = mod(J,C);
else
    mod_flg_J = 0;
end
if (ip+mod(floor(s/C),C))==C-1
    mod_flg_K = mod(K,C);
else
    mod_flg_K = 0;
end
X_sp_part = sptensor([I J K]);
X_sp_part(ip*index_part_X+1:(ip+1)*index_part_X+mod_flg_I,mod(ip+s,C)*index_part_Y+1:(mod(ip+s,C)+1)*index_part_Y+mod_flg_J,...
    (mod(ip+floor(s/C),C))*index_part_K+1:((mod(ip+floor(s/C),C))+1)*index_part_K+mod_flg_K)...
    =X(ip*index_part_X+1:(ip+1)*index_part_X+mod_flg_I,mod(ip+s,C)*index_part_Y+1:(mod(ip+s,C)+1)*index_part_Y+mod_flg_J,...
    (mod(ip+floor(s/C),C))*index_part_K+1:((mod(ip+floor(s/C),C))+1)*index_part_K+mod_flg_K);
end