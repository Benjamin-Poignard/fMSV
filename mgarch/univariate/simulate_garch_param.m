function [a,b] = simulate_garch_param(N)

gamma = true;
while(gamma)
    b = 0.7 + (0.9-0.7)*rand(1,N);
    a = 0.05 + (0.15-0.05)*rand(1,N);
    gamma = (any(b+a > 1));
end