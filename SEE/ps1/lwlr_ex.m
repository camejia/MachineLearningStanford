cd('C:\Users\Chris\Documents\GitHub\MachineLearningStanford\SEE\ps1');

[X, y] = load_data;

% tau = 0.01;
% tau = 0.05
% tau = 0.1;
tau = 0.5;
% tau = 1.0;
% tau = 5.0;

% res = 50;
res = 200;

plot_lwlr(X, y, tau, res);  % Calls lwlr
