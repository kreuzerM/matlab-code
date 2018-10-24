function [Diff,CDR] = getDiffCDR(xn,cfg)
    for q= 1:cfg.n_src
        for n = 1:cfg.n_array
        [Diff(:,n,q),CDR(:,n,q)] = compute_diff_binwise(squeeze(xn(:,:,q,n)),cfg);
        end
    end
end

