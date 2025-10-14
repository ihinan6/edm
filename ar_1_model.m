rng(123);

c = 0;               % constant
phi = 0.5;           % true coefficient/jacobian for ar(1) model 
dt = 1;              % time step
tmax = 200;          
time = 0:dt:tmax;
T = numel(time);
E = 1;          % embedding dimension
theta = 0;      % nonlinearity parameter

sigma_values = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2];  % noise levels for testing

for s = 1:length(sigma_values)
    sigma = sigma_values(s);

    % ar(1) simulation
    x = zeros(1, T);
    x(1) = 1;

    for t = 2:T
        x(t) = c + phi * x(t-1) + sigma * randn();
    end

    % s-map jacobian/coefficient estimation
    y = x;
    n = length(y);
    X = y(1:end-1)';
    jacobians = zeros(n-E, E);
    intercepts = zeros(n-E, 1);

    for t_pred = E:n-1
        X_pred = X(t_pred - (E-1), :);
        neighbor_idx = setdiff(1:size(X,1), t_pred - (E-1));
        X_neighbors = X(neighbor_idx, :);
        y_neighbors = y(neighbor_idx + E)';

        distances = sqrt(sum((X_neighbors - X_pred).^2, 2));
        weights = exp(-theta * distances / mean(distances));
        weights = weights / sum(weights);

        X_mat = [ones(size(X_neighbors,1),1), X_neighbors];
        W = diag(weights);
        a = (X_mat' * W * X_mat) \ (X_mat' * W * y_neighbors);

        intercepts(t_pred - (E-1)) = a(1);
        jacobians(t_pred - (E-1), :) = a(2:end);
    end

    % figures
    figure;
    subplot(2,1,1);
    plot(time, x, 'b', 'LineWidth', 1.2);
    title(['AR(1) Process (\phi = ', num2str(phi), ', \sigma = ', num2str(sigma), ', dt = ', num2str(dt), ')']);
    xlabel('Time');
    ylabel('x(t)');

    subplot(2,1,2);
    hold on;
    plot(time, phi * ones(size(time)), 'w--', 'LineWidth', 1.2);
    plot(jacobians, 'r', 'LineWidth', 1.2);
    hold off;
    title('True Coefficient and S-map Jacobians');
    xlabel('Time');
    ylabel('Coefficient');
    legend('True Coefficient', 'Estimated Jacobian');
    ylim([0.3 0.6]);
    
end

%% close
close all;
clear all;
