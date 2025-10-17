rng(123);

c = 0;                   % constant
phi = [0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2, 0.1, -0.05];  % 10 coefficients (decaying values)
p = length(phi);          % AR order (10)
dt = 1;
tmax = 200;
time = 0:dt:tmax;
T = numel(time);
E = p;                    % embedding dimension = 10
theta = 0;                % nonlinearity parameter

sigma_values = [0, 1, 10];  % noise levels

for s = 1:length(sigma_values)
    sigma = sigma_values(s);

    % AR(10) simulation
    x = zeros(1, T);
    x(1:p) = randn(1, p);  % initialize first 10 values

    for t = (p+1):T
        x(t) = c + phi * x(t-1:-1:t-p)' + sigma * randn();
    end

    % s-map Jacobian / coefficient estimation
    y = x;
    n = length(y);
    X = zeros(n - p, p);
    for i = 1:(n - p)
        X(i, :) = y(i+p-1:-1:i);
    end

    jacobians = zeros(n - p, p);
    intercepts = zeros(n - p, 1);

    for t_pred = 1:(n - p)
        X_pred = X(t_pred, :);
        neighbor_idx = setdiff(1:size(X,1), t_pred);
        X_neighbors = X(neighbor_idx, :);
        y_neighbors = y(neighbor_idx + p)';

        distances = sqrt(sum((X_neighbors - X_pred).^2, 2));
        weights = exp(-theta * distances / mean(distances));
        weights = weights / sum(weights);

        X_mat = [ones(size(X_neighbors,1),1), X_neighbors];
        W = diag(weights);
        a = (X_mat' * W * X_mat) \ (X_mat' * W * y_neighbors);

        intercepts(t_pred) = a(1);
        jacobians(t_pred, :) = a(2:end);
    end

    % figures
    figure;
  
    subplot(2,1,1);
    plot(time, x, 'b', 'LineWidth', 1.2);
    title(['AR(10) Process, \sigma = ', num2str(sigma)]);
    xlabel('Time');
    ylabel('x(t)');
    
    % plot true and estimated coefficients
    subplot(2,1,2);
    hold on;
    
    % plot estimated Jacobians and store handles
    h_est = gobjects(p, 1);
    for k = 1:p
        h_est(k) = plot(jacobians(:, k), 'LineWidth', 1.2, 'DisplayName', ['\phi_', num2str(k)]);
    end
    
    % plot true coefficients 
    for k = 1:p
        plot(1:length(jacobians), phi(k) * ones(size(jacobians,1),1), ...
         '--', 'Color', h_est(k).Color, 'HandleVisibility', 'off');
    end
    
    
    hold off;
    title('True Coefficients vs Estimated Jacobians');
    xlabel('Time');
    ylabel('Coefficient');
    legend('show', 'Location', 'bestoutside');

end


%% close
close all;
