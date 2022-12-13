function [U0, V0, W0, T, i_prox, EpochTime, Loss_real_for_prox_steps] = GeneralProxCPAlgorithm...
    (flg_Img, Data_file, flg_rand_state, R, q, IT, g, ITperPStep, z_perc,...
    precision,step_reduction_prc, step_change_threshhold)

% This function is implementing the Proximal Stochatic Gradient Descent
% Method (not parallel) v1.1 from the paper 
% 
% 
% T. Papastergiou and V. Megalooikonomou, "A distributed proximal gradient descent method for tensor completion", 
% 2017 IEEE International Conference on Big Data (Big Data), Boston, MA, 2017, pp. 2056-2065, doi: 10.1109/BigData.2017.8258152
% 

% This method computes correctly the Gradients (with the regularization
% terms). For this purpose the Nonzero elements of the sparse Tensor X are
% calculated once at the begining of the method. This is a costly
% calculation and it seems that the gain is little. I have to evaluate this
% procedure in contrast with the computation of the Gradients with out the
% number of Nonzero elements.
%
% flg_Img: if 1 then the data file is a image, if 2 the file is a hyperspectral image
%       else the data file is a .mat file, that must contain a tensor named X
% Data_file: the full path to the file.
% flg_rand_state: if flg_rand_state>=0  rand state is put to flg_rand_state,
%       if flg_rand_state == -1 then no rand state is used
% R: the rank of the PARAFAC Decomposition
% q: the exponent in the learning rate. the learning rate is calculated as
%       follows: 1/(2^q)
% IT: the number of maximum iterations of the method
% g: the proximal coefficient, see the paper
% ITperPStep: SGD Iterations per proximal step, see the paper
% z_perc: the percentage of the missing values (applicable only in images
%       and hyperspectral images
% precision: termination criterion of the algorithm
% step_reduction_prc -> ItPerProximalStep reduction percentage
% step_change_threshhold -> the accuracy of
% (loss(iProx)-loss(iProx-1))/loss(iProx) for ItPerProximalStep reduction

if flg_rand_state>=0
    rand('state',flg_rand_state);
end

disp('loading and preprocessing data');

if flg_Img == 1
    [I, J, K, X, IndexOfZeros] = f_imgReadAndPreprossecing(Data_file,z_perc);
elseif flg_Img == 2
    [X, IndexOfZeros, I, J, K] = f_specImgPrepross(Data_file, 500, 650, 150, 300, z_perc);
else
    load(Data_file);
    f = X.size;
    I = f(1);
    J = f(2);
    K = f(3);
end

%computation of zero elements and nonzero elements of the tensor
Nzer = nnz(X);
%create NNZ_i, NNZ_j, NNZ_k
[NNZ_i,NNZ_j,NNZ_k] = f_calculate_NNZ_i_v2_1(X);

execution_time = tic;

h = 1/(2^q);

U0 = rand(I,R);
V0 = rand(J,R);
W0 = rand(K,R);

x_subs = X.subs;
x_vals0 = X.vals;

Loss_real_for_prox_steps = zeros(1,IT);

EpochTime = zeros(1,IT);

if(flg_Img==1)
    output_fig = figure('Name','kruskal tensor output image','NumberTitle','off');
    title('ongoining kruskal tensor output image');
    loss_plot_fig = figure('Name','loss plot figure','NumberTitle','off');
    title('ongoining loss plot');
end

for i_prox = 1:IT
    disp('epoch ');
    disp(i_prox);
    disp('itPerPrStep = ');
    disp(ITperPStep);
    
    [U,V,W] = f_SGDT_reg_forProximal_Itter_v1_0_noLossCom...
        (NNZ_i,NNZ_j,NNZ_k,ITperPStep,x_vals0,x_subs,Nzer,U0,V0,W0,h,g);
    
    %real loss computation
    
    Loss_real = f_real_Tensor_LossComSGDT(X,U0,V0,W0,IndexOfZeros);    
    Loss_real_for_prox_steps(i_prox) = Loss_real;
    
    U0 = U;
    V0 = V;
    W0 = W;
    
    if Loss_real<=precision
        EpochTime(i_prox) = toc(execution_time);
        break;
    end
    
    %here we clclulate the it_per_Proximal_step taken in each itteration
    if i_prox>=2
        ITperPStep = f_it_per_proStep_manipulation_v1...
    (ITperPStep,Loss_real,Loss_real_for_prox_steps(i_prox-1), step_reduction_prc, step_change_threshhold);
    end

    
    if (flg_Img==1)
        figure(output_fig);
        kX = ktensor({U0,V0,W0});
        X_ten = tensor(kX);
        X_ten = double(X_ten);
        imshow(X_ten);
        figure(loss_plot_fig);
        plot(Loss_real_for_prox_steps);
    end
        
    EpochTime(i_prox) = toc(execution_time);
end

T = toc(execution_time);

figure('Name','Loss Plot','NumberTitle','off');

title(['r=',num2str(R),'  h=', num2str(h), '  ITperStep=', num2str(ITperPStep),...
    '  g=',num2str(g)]);
plot(Loss_real_for_prox_steps,'r');

%ploting the loss versus the wallclock time elapsed
figure('Name','Loss vs Time','NumberTitle','off');

title(['r=',num2str(R),'  h=', num2str(h), '  ITperStep=', num2str(ITperPStep),...
    '  g=',num2str(g)]);
plot(EpochTime,Loss_real_for_prox_steps,'r'); 


%output image

if flg_Img == 1
    f_imgDisplay(U0, V0, W0);
end

if flg_Img == 2
    f_spec_ImgDisp(U0, V0, W0, 63, 52, 36);
end

disp(T);
end