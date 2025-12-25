function [logp,gams,xisum] = group_runFB(mmhat, Data_fit)
    
    clear FB;
    ld = length(Data_fit);
    for ll = 1:ld
        [logp,gams,xisum] = runFB_GLMHMM(mmhat,Data(ii).xx, Data(ii).yy, Data(ii).mask);
        FB(ll).
    end

end