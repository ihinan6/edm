
rng(123);

c = 0;                    % constant
phi = [0.9, -0.8, 0.7, -0.6, 0.5];  % p coefficients (decaying values)
p = length(phi);          % AR order (p)
dt = 1;
tmax = 200;
time = 0:dt:tmax;
T = numel(time);
E = p;                    % embedding dimension = p
theta = 1;                % nonlinearity parameter

sigma_values = [0, 1, 5, 8, 10, 14, 16, 20, 25, 30];  % noise levels

for s = 1:length(sigma_values)
    sigma = sigma_values(s);

    % AR(10) simulation
    x = zeros(1, T);
    x(1:p) = randn(1, p);  % initialize first 10 values

    for t = (p+1):T
        x(t) = c + phi * x(t-1:-1:t-p)' + sigma * randn();
    end

    % s-map Jacobian / coefficient estimation
    y1 = x(p+1:end);        % one-step-ahead targets
    y2 = x(p+2:end);        % two-step-ahead targets (shorter)
    n = length(x);
    X = zeros(n - p, p);
    for i = 1:(n - p)
        X(i, :) = x(i+p-1:-1:i);
    end
    
    jacobians_1step = zeros(n - p, p);
    jacobians_2step = zeros(n - p - 1, p);  % shorter for t+2
    intercepts_1step = zeros(n - p, 1);
    intercepts_2step = zeros(n - p - 1, 1);
    
    for t_pred = 1:(n - p)
        X_pred = X(t_pred, :);
        neighbor_idx = setdiff(1:size(X,1), t_pred);
        
        % Restrict neighbors to valid y2 range (since y2 is shorter)
        neighbor_idx_2 = neighbor_idx(neighbor_idx <= length(y2));
        
        X_neighbors = X(neighbor_idx, :);
        X_neighbors_2 = X(neighbor_idx_2, :);
        y_neighbors_1 = y1(neighbor_idx)';
        y_neighbors_2 = y2(neighbor_idx_2)';  % now safe
    
        % Compute distances and weights for t_pred
        distances = sqrt(sum((X_neighbors - X_pred).^2, 2));
        weights = exp(-theta * distances / mean(distances));
        weights = weights / sum(weights);
        
        % Weighted regression for 1-step
        X_mat = [ones(size(X_neighbors,1),1), X_neighbors];
        W = diag(weights);
        a1 = (X_mat' * W * X_mat) \ (X_mat' * W * y_neighbors_1);
        intercepts_1step(t_pred) = a1(1);
        jacobians_1step(t_pred, :) = a1(2:end);
        
        % Weighted regression for 2-step
        if t_pred <= length(y2)
            X_mat_2 = [ones(size(X_neighbors_2,1),1), X_neighbors_2];
            W2 = diag(weights(1:length(neighbor_idx_2)));
            a2 = (X_mat_2' * W2 * X_mat_2) \ (X_mat_2' * W2 * y_neighbors_2);
            intercepts_2step(t_pred) = a2(1);
            jacobians_2step(t_pred, :) = a2(2:end);
        end
    end

    figure;
    
    % ---- Panel 1: Simulated data ----
    subplot(3,1,1);
    plot(time, x, 'b', 'LineWidth', 1.2);
    title(['AR(', num2str(p), ') Process, \sigma = ', num2str(sigma)]);
    xlabel('Time');
    ylabel('x(t)');
    ylim([-100 100]);
    grid on;
    
    % ---- Panel 2: Estimated 1-step Jacobians ----
    subplot(3,1,2);
    hold on;
    h1 = gobjects(p,1);
    for k = 1:p
        h1(k) = plot(jacobians_1step(:,k), 'LineWidth', 1.2, 'DisplayName', ['1-step \phi_', num2str(k)]);
        % Overlaying true phi values as dashed lines
        plot(phi(k) * ones(size(jacobians_1step,1),1), '--', 'Color', h1(k).Color, 'HandleVisibility', 'off');
    end
    hold off;
    title('Estimated 1-step Jacobians vs True Coefficients');
    xlabel('Time Index');
    ylabel('Coefficient');
    legend('show', 'Location', 'bestoutside');
    grid on;
    
    % ---- Panel 3: Estimated 2-step Jacobians ----
    subplot(3,1,3);
    hold on;
    h2 = gobjects(p,1);
    for k = 1:p
        h2(k) = plot(jacobians_2step(:,k), 'LineWidth', 1.2, 'DisplayName', ['2-step \phi_', num2str(k)]);
        % Overlay true phi values again
        plot(phi(k).^2 * ones(size(jacobians_2step,1),1), '--', 'Color', h2(k).Color, 'HandleVisibility', 'off');
    end
    hold off;
    title('Estimated 2-step Jacobians vs True Coefficients');
    xlabel('Time Index');
    ylabel('Coefficient');
    legend('show', 'Location', 'bestoutside');
    grid on;
    
    % figure;
    % 
    % subplot(2,1,1);
    % hold on;
    % for k = 1:p
    %     plot(jacobians_1step(:,k), 'LineWidth', 1.2, 'DisplayName', ['1-step \phi_', num2str(k)]);
    % end
    % title('Estimated 1-step Jacobians');
    % xlabel('Time'); ylabel('Coefficient');
    % legend('show', 'Location', 'bestoutside');
    % hold off;
    % 
    % subplot(2,1,2);
    % hold on;
    % for k = 1:p
    %     plot(jacobians_2step(:,k), 'LineWidth', 1.2, 'DisplayName', ['2-step \phi_', num2str(k)]);
    % end
    % title('Estimated 2-step Jacobians');
    % xlabel('Time'); ylabel('Coefficient');
    % legend('show', 'Location', 'bestoutside');
    % hold off;


end

disp(numel(jacobians));


%% close
close all;
