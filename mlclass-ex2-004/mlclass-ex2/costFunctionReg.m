function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%{
% Cost Function
z = X*theta;
hFunc = sigmoid(z);
cost1 = -(y.*log(hFunc));
cost0 = -((1-y).*log(1-hFunc));
reg = (lambda/(2*m))*sum(theta.^2);
J = sum(cost1 + cost0)/m + reg;

% Gradient
gradFunc = (hFunc-y)'*X;
grad = gradFunc/m;
%}

n = length(theta);
% Caculate J
sum1 = 0;
for i=1:m
  tempz = X(i, :)*theta;
  tempg = sigmoid(tempz);
  temp1 = -(y(i)*log(tempg));
  temp0 = -((1-y(i))*log(1-tempg));  
  sum1 = sum1 + temp1 + temp0;  
end

regTemp = 0;
for j=2:n
  regTemp = regTemp + theta(j)^2;
end
reg = (lambda/(2*m))*regTemp;

J = sum1/m + reg;

% Caculate grad
sumTemp1 = 0;
for i=1:m
  tempz = X(i, :)*theta;
  tempg = sigmoid(tempz);
  sumTemp1 = sumTemp1+(tempg-y(i))*X(i, 1);
end
grad(1) = sumTemp1/m;

for j=2:n
  sumTemp = 0;
  for i=1:m
    tempz = X(i, :)*theta;
    tempg = sigmoid(tempz);
    sumTemp = sumTemp+(tempg-y(i))*X(i, j);
  end
  reg = (lambda/m)*theta(j);
  grad(j) = sumTemp/m + reg;
end



% =============================================================

end
