function [Loss_per_IT_Tensor_real] = f_real_Tensor_LossComSGDT(X,U0,V0,W0,IndexOfZeros)

%This function calculates the Real Relative Loss of the tensor for each
%epoch. The loss is computed in
%terms of the training Tensor (i.e. the missing values are ignored). The
%loss is computed by Training_ternsor - Computed_Tensor / Training_tensor
%regarding only the non missing values of either the Training_tensor and
%the Computed_Tensor. Values in the Computed_Tensor that have indices of
%the missing values are ignored.
%
%The computation is performed using kruskal and sparse tesnors and performs
%better that the f_realLossComSGDT function that calculates the same thing
%in an elementwise manner
%
%Input arguments
%X = the sparse tensor (must be of type sptensor) with the missing values
%ignored
%U0,V0,W0 = the computed PARAFAC factors
%IndexOfZeros = the indices of the nonzero elements of the initial tensor
    
kX = ktensor({U0,V0,W0});
    
kX_ten = tensor(kX);
kX_ten(IndexOfZeros') = 0;

Loss_per_IT_Tensor_real = norm(kX_ten-X)/norm(X); %real loss
    
    
    
end
