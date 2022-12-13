function [] = f_spec_ImgDisp(U, V, W, Rspec, Gspec, Bspec)

W_R = W(Rspec, :);
W_G = W(Gspec, :);
W_B = W(Bspec, :);

W_img = zeros(3, length(W_R));

W_img(1,:) = W_R;
W_img(2,:) = W_G;
W_img(3,:) = W_B;

kX = ktensor({U,V,W_img});
X_ten = tensor(kX);
X_ten = double(X_ten);
figure;
imshow(X_ten);

end