function [new_IT] = f_it_per_proStep_manipulation_v1...
    (IT,loss_i,loss_i_1, step_reduction_prc, step_change_threshhold)

%this function manipulates the number of itterations of SGD made in each
%proximal step. (Loss(i_prox-1) - Loss(i_prox))/Loss(i_prox). IT must be
%greater or equal than 2


if ( abs((loss_i-loss_i_1)/loss_i)<step_change_threshhold ) && (IT>5)
    new_IT = ceil(IT*step_reduction_prc);
else
    new_IT = IT;
end

end
