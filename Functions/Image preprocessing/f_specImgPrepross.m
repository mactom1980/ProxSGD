function [X,IndexOfZeros, I, J, K] = f_specImgPrepross(img_file, x_str, x_end, y_str, y_end,z_perc)

X = imread(img_file);

X = im2double(X);

X = X(x_str:x_end, y_str:y_end, :);

[I, J, K] = size(X);

X = tensor(X, [I J K]);

% i replace the zero elements of the image
% by 0.00001

[x_1,~] = find(X==0);
if ~isempty(x_1)
    X(x_1) = .00001;
end

% the sparse tensor
Nel = I*J*K;


Zel = round(Nel*z_perc);
% Nzer = Nel-Zel;

IndexOfZeros = randperm(Nel,Zel);
X(IndexOfZeros') = 0;

A = double(X);

A_img = zeros(I,J,3);
A_img(:,:,1) = A(:,:,63);
A_img(:,:,2) = A(:,:,52);
A_img(:,:,3) = A(:,:,36);

figure('Name','Input image','NumberTitle','off')
title(['Missing elements percentage ', num2str(z_perc)]) 
imshow(A_img)


X = sptensor(X);

end