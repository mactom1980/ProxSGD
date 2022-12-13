function [NNZ_i,NNZ_j,NNZ_k] = f_calculate_NNZ_i_v2_1(X)

%this function computes the NNZ_i's (i.e. the nonzero elements of each
%slice) of a loaded tensor X. The calculation is performed with the use of
%findInSorted.m file rettrieved in
%https://github.com/danielroeske/danielsmatlabtools/blob/master/matlab/data/findinsorted.m.
%this function performs a binary search in A SORTED array and returns the
%range a:b in which x(a)==x(b).
%The calculation is perforemd in parallel.
%Data_file -> is the .mat file, where the tensor is located. The tensor
%must be saved with the name X and must be a sparse tensor.
%The NNZ_i's arrays are saved in different .mat files in the current
%folder.




f = X.size;
I = f(1);
J = f(2);
K = f(3);
NNZ_i = zeros(I,1);
NNZ_j = zeros(J,1);
NNZ_k = zeros(K,1);


Xsubs_i = sort(X.subs(:,1));
Xsubs_j = sort(X.subs(:,2));
Xsubs_k = sort(X.subs(:,3));



disp('calculating NNZ_i')
tic;
x_i_uni = unique(Xsubs_i);
tmp_NNZ_i = zeros(length(x_i_uni),1);

for i=1:length(x_i_uni)
    [a,b] = findInSorted(Xsubs_i,x_i_uni(i));
    tmp_NNZ_i(i) = b-a+1;
end
 
NNZ_i(x_i_uni) = tmp_NNZ_i;

T=toc;
disp(['elapsed time:', num2str(T)]);

clear tmp_NNZ_i;

clear x_i_uni;

disp('calculating NNZ_j');
tic;
x_j_uni = unique(Xsubs_j);
tmp_NNZ_j = zeros(length(x_j_uni),1);

for i=1:length(x_j_uni)
    [a,b] = findInSorted(Xsubs_j,x_j_uni(i));
    tmp_NNZ_j(i) = b-a+1;   
end

NNZ_j(x_j_uni) = tmp_NNZ_j;

T=toc;
disp(['elapsed time:', num2str(T)]);

clear x_j_uni;
clear tmp_NNZ_j;

disp('calculating NNZ_K');

tic;
x_k_uni = unique(Xsubs_k);
tmp_NNZ_k = zeros(length(x_k_uni),1);

for i=1:length(x_k_uni)
    [a,b] = findInSorted(Xsubs_k,x_k_uni(i));
    tmp_NNZ_k(i) = b-a+1;
end

NNZ_k(x_k_uni) = tmp_NNZ_k;

T=toc;
disp(['elapsed time:', num2str(T)]);

clear x_k_uni;
clear tmp_NNZ_k;

end