function g = Grad(fun, x0)
% |delta(i)| is relative to |x0(i)|
delta = x0 / 1000;             
for i = 1 : length(x0)
    if x0(i) == 0
          % avoids delta(i) = 0 (**arbitrary value**)
        delta(i) = 1e-12;      
    end
     % recovers original x0
    u = x0;                    
    u(i) = x0(i) + delta(i);
     % fun(x0(i-1), x0(i)+delta(i), x0(i+1), ...)
    f1 = feval ( fun, u );     
    u(i) = x0(i) - delta(i);
     % fun(x0(i-1), x0(i)-delta(i), x0(i+1), ...)
    f2 = feval ( fun, u );     
   
     % partial derivatives in column vector
    g(i,1) = (f1 - f2) / (2 * delta(i)); 
end