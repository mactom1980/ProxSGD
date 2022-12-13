function [U0, V0, W0, T, i_prox, EpochTime, Loss_real_for_prox_steps] = GeneralProxCPAlgorithm_v0_frailSafe...
    (flg_Img, Data_file, flg_rand_state, R, q, IT, g, ITperPStep, z_perc,...
    precision,tol, step_reduction_prc, step_change_threshhold)

%This function is implementing the Proximal Stochatic Gradient Descent
%Method (not parallel) v1.1. It was similar to SGDT_Img_com_v1_0_with_reg.
%This method computes correctly the Gradients (with the regularization
%terms). For this purpose the Nonzero elements of the sparse Tensor X are
%calculated one's at the begining of the method. This is a costly
%calculation and it seems that the gain is little. I have to evaluate this
%procedure in contrast with the computation of the Gradients with out the
%number of Nonzero elements.
%This method uses the f_SGT_reg_forProximal_Itter_v_1_0.
%The other method (without the number of nonzero elements) will be
%implemented in SGDT_Img_com_v1_0_with_reg.
%This method does not currently handle images with zero entries.
%
%flg_Img: Image Flag -> 1 regular image, 2-> spectral image, else datafile
%step_reduction_prc -> ItPerProximalStep reduction percentage
%step_change_threshhold -> the accuracy of
%(loss(iProx)-loss(iProx-1))/loss(iProx) for ItPerProximalStep reduction

%If you use the GenrProxSGD algorithm please site:

%Papastergiou, T. and V. Megalooikonomou. A distributed proximal gradient descent method for tensor completion.
% in 2017 IEEE International Conference on Big Data (Big Data). 2017.

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

%comutation of zero elements and nonzero elements of the tensor
% Nel = I*J*K;
% Zel = round(Nel*z_perc);
Nzer = nnz(X);
norm_X = norm(X);

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
    
    Loss_real = f_compute_real_Loss_StrSGD(U0,V0,W0,X);
    Loss_real = sqrt(Loss_real)/norm_X;
    disp(['loss: ' num2str(Loss_real)])
    Loss_real_for_prox_steps(i_prox) = Loss_real;
    
%     Loss_real_for_prox_steps(i_prox) = Loss_real;
    
    U0 = U;
    V0 = V;
    W0 = W;
    
    
    
    %here we clclulate the it_per_Proximal_step taken in each itteration
    if i_prox>=2
%         IT = f_it_per_proStep_manipulation(IT,Loss_real_for_prox_steps,i_prox);
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
    
    if Loss_real<=precision
        EpochTime(i_prox) = toc(execution_time);
        break;
    end
    
     if  (i_prox>=2) && abs(Loss_real_for_prox_steps(i_prox) - Loss_real_for_prox_steps(i_prox-1)) < tol
        EpochTime(i_prox) = toc(execution_time);
        disp('The method converged!!!')
        break;
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