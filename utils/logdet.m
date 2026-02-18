function iC = logdet(mA) %#codegen
    iC = 2 * sum(log(diag(chol(mA))));
end