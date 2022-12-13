function [ii] = f_convert_iIndex(i, C, I)

f = floor(I/C);

if i>= (I - (mod(I,C)-1))
    flg_mod = f;
else
    flg_mod = 0;
end

ii = i - floor((i-1)/f)*f + flg_mod;

end