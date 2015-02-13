function y = lwlr(X_train, y_train, x, tau)

lambda = 0.0001;  % regularization parameter

m = size(X_train, 1);
n = size(X_train, 2);

% theta includes the intercept term
theta = zeros(n + 1, 1);
X_aug = [ones(m, 1), X_train];

w =  bsxfun(@minus, x.', X_train);
w = sum(w .* w, 2);
w = exp(-w / (2 * tau * tau));

for iter = 1 : 10
    h = X_aug * theta;
    h = 1.0 ./ (1.0 + exp(-h));
    
    z = w .* (y_train - h);
    
    % gradient
    grad = X_aug.' * z - lambda * theta;
    
    % Hessian
    D = diag(-w .* h .* (1.0 - h));
    H = X_aug.' * D * X_aug - lambda * eye(n + 1);
    
    theta = theta - H \ grad;
end

h = X_aug * theta;
h = 1.0 ./ (1.0 + exp(-h));
if (h > 0.5)
    y = 1;
else
    y = 0;
end

end
