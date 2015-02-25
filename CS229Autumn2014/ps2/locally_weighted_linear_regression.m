function y = locally_weighted_linear_regression(X_train, y_train, x, tau)

m = size(X_train, 1);
n = size(X_train, 2);

np = n + 1;
X_aug = [ones(m, 1) X_train];
theta = zeros(np, 1);

w =  bsxfun(@minus, x.', X_train);
w = sum(w .* w, 2);
w = exp(-w / (2 * tau * tau));

theta = bsxfun(@times, w, X_aug) \ (w .* y_train);

y = [1 x] * theta;

end
