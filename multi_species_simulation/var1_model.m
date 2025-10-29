rng(123);

% parameters
T = 100;
A = [0.1 0.4 0.1 0.2;
     0.1 -0.2 0.3 0.2;
     0.8 0.2 0.1 0.2;
     0.1 0.2 0.3 0.1];
[row, col] = size(A);
sigma_values = [0, 0.1, 1, 4, 8];  % noise levels
disp(det(A));

theta = 1;   % nonlinearity weighting
E = col;     % embedding dimension = number of species


for s = 1:length(sigma_values)
    sigma = sigma_values(s);

    % simulating VAR(1)
    X = zeros(T, col);
    for t = 1:T-1
        X(t+1,:) = (A * X(t,:)')' + sigma * randn(1, col);
    end

    % s-map estimation (k=1)
    jacobian_est_k1 = zeros(T-E, col, col);
    for dim = 1:col
        for t_pred = 1:T-E
            X_pred = X(t_pred, :);
            neighbor_idx = setdiff(1:T-E, t_pred);
            X_neighbors = X(neighbor_idx, :);
            y_neighbors = X(E + neighbor_idx, dim);

            % distances and weights
            distances = sqrt(sum((X_neighbors - X_pred).^2, 2));
            weights = exp(-theta * distances / mean(distances));
            weights = weights / sum(weights);

            % weighted regression
            X_mat = [ones(length(neighbor_idx),1), X_neighbors];
            W = diag(weights);
            a = (X_mat' * W * X_mat) \ (X_mat' * W * y_neighbors);
            jacobian_est_k1(t_pred, dim, :) = a(2:end);
        end
    end

    % determinant of estimated Jacobians (k=1)
    J_det_est_k1 = zeros(T-E,1);
    for t = 1:T-E
        J_det_est_k1(t) = det(squeeze(jacobian_est_k1(t,:,:)));
    end
    J_det_true_k1 = det(A) * ones(T,1);

    % s-map estimation (k=2)
    jacobian_est_k2 = zeros(T-2*E, col, col);
    for dim = 1:col
        for t_pred = 1:T-2*E
            X_pred = X(t_pred, :);
            neighbor_idx = setdiff(1:T-2*E, t_pred);
            X_neighbors = X(neighbor_idx, :);
            y_neighbors = X(E*2 + neighbor_idx, dim);

            % distances and weights
            distances = sqrt(sum((X_neighbors - X_pred).^2, 2));
            weights = exp(-theta * distances / mean(distances));
            weights = weights / sum(weights);

            % weighted regression
            X_mat = [ones(length(neighbor_idx),1), X_neighbors];
            W = diag(weights);
            a = (X_mat' * W * X_mat) \ (X_mat' * W * y_neighbors);
            jacobian_est_k2(t_pred, dim, :) = a(2:end);
        end
    end

    % determinant of estimated Jacobians (k=2)
    J_det_est_k2 = zeros(T-2*E,1);
    for t = 1:T-2*E
        J_det_est_k2(t) = det(squeeze(jacobian_est_k2(t,:,:)));
    end
    J_det_true_k2 = det(A^2) * ones(T,1);

    % Figure for this sigma
    figure('Name', sprintf('Noise σ = %.3f', sigma), 'NumberTitle', 'off');

    % panel 1: species data simulation
    subplot(2,1,1);
    hold on;
    for i = 1:col
        plot(1:T, X(:,i), 'LineWidth', 1.5);
    end
    xlabel('Time');
    ylabel('Population');
    title(sprintf('VAR(1) Simulation, σ = %.2f', sigma));
    legend(arrayfun(@(i) sprintf('Species %d', i), 1:col, 'UniformOutput', false));
    grid on; hold off;
    yline([-10,20]);

    % panel 2: Jacobian determinants (k=1 vs k=2), true and estimated
    subplot(2,1,2);
    hold on;
    plot(1:T, J_det_true_k1, 'r--', 'LineWidth', 1.5);
    plot(1:T, J_det_true_k2, 'b--', 'LineWidth', 1.5);
    plot(E+1:T, J_det_est_k1, 'r', 'LineWidth', 1.2);
    plot(2*E+1:T, J_det_est_k2, 'b', 'LineWidth', 1.2);
    xlabel('Time');
    ylabel('Jacobian Determinant');
    title(sprintf('True vs Estimated Jacobians (σ = %.2f)', sigma));
    legend({'True det(A)', 'True det(A^2)', 'S-map (k=1)', 'S-map (k=2)'}, 'Location', 'best');
    grid on; hold off;
end


%% clear figures
close all;