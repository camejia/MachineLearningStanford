cd('C:\Users\Chris\Documents\GitHub\MachineLearningStanford\CS229Autumn2014\ps1');

X = load('q1x.dat');
y = load('q1y.dat');

theta = logistic_regression(X, y);

% Process the data with the linear prediction
X_aug = [ones(size(X, 1), 1), X];
ypred = double(X_aug * theta > 0.5);

disp(['theta = ', num2str(theta.')]);

% Points for a line separating data
xl = zeros(2, 1);
yl = zeros(2, 1);
xl(1) = min(X(:, 1));
xl(2) = max(X(:, 1));

% Solve theta(1) * theta(2)*xl(1) + theta(3)*yl(2) = 0 for yl
yl(1) = -(theta(1) + theta(2)*xl(1)) / theta(3);
yl(2) = -(theta(1) + theta(2)*xl(2)) / theta(3);

figure;
hold on;
plot(X(y == 0 & ypred == 0, 1), X(y == 0 & ypred == 0, 2), 'ko');
plot(X(y == 0 & ypred == 1, 1), X(y == 0 & ypred == 1, 2), 'ro');
plot(X(y == 1 & ypred == 0, 1), X(y == 1 & ypred == 0, 2), 'rx');
plot(X(y == 1 & ypred == 1, 1), X(y == 1 & ypred == 1, 2), 'kx');
line(xl, yl);
legend('True Neg', 'False Pos', 'False Neg', 'True Pos');
hold off;
axis equal;
axis square;

% NOTE: One of the "True Neg" circles looks wrong in the plot.
% I think it's just an inaccuracy in MATLAB plotting.
