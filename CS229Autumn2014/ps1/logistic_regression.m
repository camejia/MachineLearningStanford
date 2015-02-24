function theta = logistic_regression(X_train, y_train, x)

m = size(X_train, 1);
n = size(X_train, 2);

np = n + 1;
X_aug = [ones(m, 1) X_train];

theta = zeros(np, 1);

grad = ones(n, 1);
while norm(grad) > 1e-8
    h = X_aug * theta;
    h = 1.0 ./ (1.0 + exp(-h));
    
    z = y_train - h;
    
    % gradient
    grad = X_aug.' * z;
    
    % Hessian
    D = diag(-h .* (1.0 - h));
    H = X_aug.' * D * X_aug;
    
    theta = theta - H \ grad;
end

end
