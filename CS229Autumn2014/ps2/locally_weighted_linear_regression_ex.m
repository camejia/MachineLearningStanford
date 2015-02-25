clear all;
close all;

x_train = load('q2x.dat');
y_train = load('q2y.dat');

m = size(x_train, 1);

X_aug = [ones(m, 1) x_train];
theta = X_aug \ y_train;

y_pred = X_aug * theta;

figure;
plot(x_train, y_train, '.', ...
    x_train, y_pred, '-');

tau = 0.8;  % looks pretty good!
% tau = 0.1;  % way overfits
% tau = 0.3;  % slightly overfit
% tau = 2.0;  % a bit too much smoothing?
% tau = 10.0;  % underfit, almost back to unweighted

x_test = linspace(min(x_train), max(x_train));
y_test = zeros(size(x_test));

for ind = 1 : length(x_test)
    y_test(ind) = locally_weighted_linear_regression(x_train, y_train, ...
        x_test(ind), tau);
end

figure;
plot(x_train, y_train, '.', ...
    x_test, y_test, '-');
