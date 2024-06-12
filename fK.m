function iK = fK(x)
  if ((x>=0)&&(x<0.5))
      iK=1-6*(x^2)+6*(x^3);
  elseif ((x>=0.5)&&(x<=1))
      iK=2*((1-x)^3);
  elseif (x>1)
      iK=0;
  end
end