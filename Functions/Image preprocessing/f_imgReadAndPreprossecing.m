function [I, J, K, X,IndexOfZeros] = f_imgReadAndPreprossecing(Data_file,z_perc)

%this function implements the preprocessing of the image file.

% reading the image
A = imread(Data_file);
A = im2double(A);

[I, J, K] = size(A);

X = tensor(A, [I J K]);
 
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
figure('Name','Input image','NumberTitle','off')
 
imshow(A)
title(['Missing elements percentage ', num2str(z_perc)])

X = sptensor(X);

end