function [] = f_imgDisplay(U0_new, V0_new,W0_new)

figure('Name','kruskal tensor output image','NumberTitle','off');
title('kruskal tensor output image');

kX = ktensor({U0_new,V0_new,W0_new});
X_ten = tensor(kX);
X_ten = double(X_ten);
imshow(X_ten);

end