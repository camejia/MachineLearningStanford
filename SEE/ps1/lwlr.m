function y = lwlr(X_train, y_train, x, tau)

lambda = 0.0001;  % regularization parameter

m = size(X_train, 1);
n = size(X_train, 2);

% Not sure whether we need intercept term
intercept_term = true;
if intercept_term
    np = n + 1;
    X_aug = [ones(m, 1) X_train];
else
    np = n;
    X_aug = X_train;
end
theta = zeros(np, 1);

w =  bsxfun(@minus, x.', X_train);
w = sum(w .* w, 2);
w = exp(-w / (2 * tau * tau));
% w = exp(-w / (2 * tau));  % Incorrect implementation from solution

grad = ones(n, 1);
while norm(grad) > 1e-6
    h = X_aug * theta;
    h = 1.0 ./ (1.0 + exp(-h));
    
    z = w .* (y_train - h);
    
    % gradient
    grad = X_aug.' * z - lambda * theta;
    
    % Hessian
    D = diag(-w .* h .* (1.0 - h));
    H = X_aug.' * D * X_aug - lambda * eye(np);
    
    theta = theta - H \ grad;
end

if intercept_term
    h = [1; x].' * theta;
else
    h = x.' * theta;
end
h = 1.0 / (1.0 + exp(-h));
if h > 0.5
    y = 1;
else
    y = 0;
end

end
