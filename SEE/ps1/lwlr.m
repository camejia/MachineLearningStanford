function y = lwlr(X_train, y_train, x, tau)

lambda = 0.0001;  % regularization parameter

m = size(X_train, 1);
n = size(X_train, 2);

theta = zeros(n, 1);

w =  bsxfun(@minus, x.', X_train);
w = sum(w .* w, 2);
w = exp(-w / (2 * tau * tau));
% w = exp(-w / (2 * tau));  % Incorrect implementation from solution

grad = ones(n, 1);
while norm(grad) > 1e-6
    h = X_train * theta;
    h = 1.0 ./ (1.0 + exp(-h));
    
    z = w .* (y_train - h);
    
    % gradient
    grad = X_train.' * z - lambda * theta;
    
    % Hessian
    D = diag(-w .* h .* (1.0 - h));
    H = X_train.' * D * X_train - lambda * eye(n);
    
    theta = theta - H \ grad;
end

h = x.' * theta;
h = 1.0 / (1.0 + exp(-h));
if h > 0.5
    y = 1;
else
    y = 0;
end

end
