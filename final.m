%% FINAL INTEGRATED SCRIPT: Ship Motion Denoising & Prognosis
% Optimized for MATLAB Syntax Stability and Journal-Level Reporting
clear; clc; close all;

%%%%%%%%%%%%% PART 1: DATA PREPARATION & DENOISING %%%%%%%%%%%%%%%%%%%%%

%% 1. PARAMETERS & DATA LOADING
max_level   = 4; 
rho         = 0.1; 
stack_depth_den = 15;
win         = 600; 
overlap     = 300; 
step        = win - overlap;

mid_row_idx = ceil(stack_depth_den/2);
hankel_shift_den = floor(stack_depth_den / 2);

if ~isfile('ship_data.csv'), error('ship_data.csv not found!'); end
M = csvread('ship_data.csv', 1, 0);
time = M(:,1); dt = mean(diff(time));

states = struct();
states(1).name = 'u (Surge)';     states(1).idx = 8;  states(1).lvls = [0, 1];
states(2).name = 'v (Sway)';      states(2).idx = 9;  states(2).lvls = [0, 1, 2];
states(3).name = 'p (Roll Rate)'; states(3).idx = 10; states(3).lvls = [0, 1];
states(4).name = 'r (Yaw Rate)';  states(4).idx = 11; states(4).lvls = [0, 1];
states(5).name = 'phi (Roll)';    states(5).idx = 12; states(5).lvls = [0, 1, 2];

N = length(time);
denoised_results = zeros(N, length(states));
full_recon_results = zeros(N, length(states)); 

%% 2. DENOISING LOOP (Sections 4.1 & 4.2 Reconstruction)
for s = 1:length(states)
    fprintf('Denoising %s...\n', states(s).name);
    rec_trend = zeros(N, 1); rec_full = zeros(N, 1); weight = zeros(N, 1);
    data_in = M(:, states(s).idx);
    
    for k = 1:step:(N-win)
        idx_win = k:(k+win-1);
        mu_val = mean(data_in(idx_win)); sig_val = std(data_in(idx_win));
        if sig_val < 1e-6, sig_val = 1; end
        norm_win = (data_in(idx_win) - mu_val) / sig_val;
        
        H_den = [];
        for i = 1:stack_depth_den
            H_den = [H_den; norm_win(i:end-stack_depth_den+i)'];
        end
        
        % Core mrDMD Calculation
        tree = run_mrdmd_recursive(H_den, dt, 0, max_level, rho);
        
        % Section 4.2: Extract Maneuvering Trend (L0-L2)
        trend_norm = zeros(1, size(H_den,2));
        for lvl = states(s).lvls
            lvl_data = get_level_data(tree, lvl);
            if ~isempty(lvl_data)
                trend_norm = trend_norm + lvl_data(mid_row_idx, :);
            end
        end
        
        % Section 4.1: Extract Full Reconstruction (L0-L4)
        full_norm = zeros(1, size(H_den,2));
        for lvl = 0:max_level
            lvl_data = get_level_data(tree, lvl);
            if ~isempty(lvl_data)
                full_norm = full_norm + lvl_data(mid_row_idx, :); 
            end
        end
        
        % Convert back to physical units
        trend_phys = (trend_norm * sig_val) + mu_val;
        full_phys = (full_norm * sig_val) + mu_val;
        
        % SECURE INDEXING: Fixed the previous "Invalid expression" error
        tloc = idx_win(1:length(trend_phys)) + hankel_shift_den;
        valid = (tloc <= N);
        
        rec_trend(tloc(valid)) = rec_trend(tloc(valid)) + trend_phys(valid)';
        rec_full(tloc(valid))  = rec_full(tloc(valid)) + full_phys(valid)';
        weight(tloc(valid))    = weight(tloc(valid)) + 1;
    end
    denoised_results(:, s) = rec_trend ./ max(weight, 1);
    full_recon_results(:, s) = rec_full ./ max(weight, 1);
end

%%%%%%%%%%%%% PART 3: OPTIMIZED MONTE CARLO PREDICTION %%%%%%%%%%%%

n_iterations = 25; 
stack_depth  = 60;  % Optimized for coupled dynamics
pred_horizon = 20; 
train_win    = 300; 
X_raw = denoised_results';

iter_RMSE = zeros(n_iterations, 5); iter_NRMSE = zeros(n_iterations, 5);
iter_RelL2 = zeros(n_iterations, 5); iter_Horizon = zeros(n_iterations, 5);

fprintf('\nStarting Optimized Monte Carlo (n=%d iterations)...\n', n_iterations);
for n = 1:n_iterations
    n1 = randi([1, N - pred_horizon - train_win - 1]);
    split_idx = n1 + train_win; n2 = split_idx + pred_horizon;
    X_train = X_raw(:, n1:split_idx);
    X_true_future = X_raw(:, split_idx+1 : n2);
    
    % Multivariate Hankel-DMD
    H = [];
    for i = 1:stack_depth, H = [H; X_train(:, i : end - stack_depth + i)]; end
    X1 = H(:, 1:end-1); X2 = H(:, 2:end);
    
    [U, S, V] = svd(X1, 'econ'); 
    r = 15; % Rank Truncation to prevent overfitting
    Ur = U(:, 1:r); Sr = S(1:r, 1:r); Vr = V(:, 1:r);
    
    Atilde = Ur' * X2 * Vr / Sr; [W, D] = eig(Atilde);
    lambda = diag(D); omega = log(lambda) / dt;
    Phi = X2 * Vr / Sr * W;
    
    b_pred = Phi \ H(:, end);
    H_pred = zeros(size(H,1), pred_horizon);
    for k = 1:pred_horizon
        H_pred(:, k) = real(Phi * (exp(omega * k * dt) .* b_pred));
    end
    X_dmd_pred = H_pred(end-4:end, :);
    
    range_train = max(X_train, [], 2) - min(X_train, [], 2);
    for s = 1:5
        y_true = X_true_future(s, :); y_pred = X_dmd_pred(s, :);
        e_rmse = sqrt(mean((y_true - y_pred).^2));
        
        iter_RMSE(n, s) = e_rmse;
        iter_NRMSE(n, s) = e_rmse / (range_train(s) + 1e-9);
        iter_RelL2(n, s) = norm(y_true - y_pred) / (norm(X_train(s, :)) + 1e-9);
        
        idx_fail = find(abs(y_true - y_pred) > 0.05*range_train(s), 1, 'first');
        if isempty(idx_fail), iter_Horizon(n, s) = pred_horizon * dt;
        else, iter_Horizon(n, s) = (idx_fail - 1) * dt; end
    end
end

%%%%%%%%%%%%% PART 4: JOURNAL VISUALIZATION %%%%%%%%%%%%%%%%%%%%%

%% 4.1 RECONSTRUCTION (L0-L4)
figure('Color','w', 'Name', '4.1 Reconstruction Plots');
table1_data = cell(5, 4);
for s = 1:5
    y_raw = M(:, states(s).idx); y_rec = full_recon_results(:, s);
    rmse_v = sqrt(mean((y_raw - y_rec).^2));
    nrmse_v = rmse_v / (max(y_raw) - min(y_raw));
    r2_v = 1 - sum((y_raw - y_rec).^2) / sum((y_raw - mean(y_raw)).^2);
    table1_data(s,:) = {states(s).name, rmse_v, nrmse_v, r2_v};
    subplot(5, 1, s); plot(time, y_raw, 'Color', [0.8 0.8 0.8], 'DisplayName', 'Original (Noisy)'); hold on;
    plot(time, y_rec, 'k--', 'LineWidth', 1, 'DisplayName', 'Full Recon (L0-L4)');
    ylabel(states(s).name); grid on;
    if s==1, title('4.1 Full mrDMD Reconstruction'); legend('show','Location','best'); end
end
fprintf('\n--- Table 1: Reconstruction Metrics ---\n');
disp(cell2table(table1_data, 'VariableNames', {'State','RMSE','NRMSE','R2'}));

%% 4.2 TREND & SPECTRAL VALIDATION
figure('Color','w', 'Name', '4.2 Denoised Trend');
table2_data = cell(5, 4);
for s = 1:5
    y_raw = M(:, states(s).idx); y_trend = denoised_results(:, s);
    v_raw = var(y_raw); v_trend = var(y_trend); red = (1 - v_trend/v_raw)*100;
    table2_data(s,:) = {states(s).name, v_raw, v_trend, red};
    subplot(5, 1, s); plot(time, y_raw, 'Color', [0.8 0.8 0.8], 'DisplayName', 'Raw'); hold on;
    plot(time, y_trend, 'r', 'LineWidth', 1.2, 'DisplayName', 'Trend (L0-L2)');
    ylabel(states(s).name); grid on;
    if s==1, title('4.2 Maneuvering Trend Extraction'); legend('show','Location','best'); end
end
fprintf('\n--- Table 2: Variance Reduction ---\n');
disp(cell2table(table2_data, 'VariableNames', {'State','Var_Raw','Var_Trend','Reduction_Pct'}));

% FFT Spectral Validation
figure('Color','w', 'Name', '4.2 Spectral Validation');
fs = 1/dt;
for s = 1:5
    subplot(5, 1, s);
    y_r = M(:, states(s).idx) - mean(M(:, states(s).idx)); Nfft = length(y_r); X_r = fft(y_r);
    psd_r = (1/(fs*Nfft)) * abs(X_r(1:floor(Nfft/2)+1)).^2; psd_r(2:end-1) = 2*psd_r(2:end-1);
    y_t = denoised_results(:, s) - mean(denoised_results(:, s)); X_t = fft(y_t);
    psd_t = (1/(fs*Nfft)) * abs(X_t(1:floor(Nfft/2)+1)).^2; psd_t(2:end-1) = 2*psd_t(2:end-1);
    freq = 0:fs/Nfft:fs/2;
    semilogy(freq, psd_r, 'Color', [0.8 0.8 0.8], 'DisplayName', 'Raw Spectrum'); hold on;
    semilogy(freq, psd_t, 'r', 'DisplayName', 'Denoised'); ylabel('PSD'); grid on; xlim([0 0.5]);
    if s==1, title('4.2 Spectral Validation (FFT)'); legend('show','Location','best'); end
end

%% 4.3 PREDICTION PERFORMANCE
avg_RMSE = mean(iter_RMSE); avg_NRMSE = mean(iter_NRMSE);
avg_RelL2 = mean(iter_RelL2); avg_Horizon = mean(iter_Horizon);

fprintf('\n--- Table 3: Prediction Metrics (Averaged n=25) ---\n');
table3_data = cell(5, 5);
for s = 1:5
    table3_data(s,:) = {states(s).name, avg_RMSE(s), avg_NRMSE(s), avg_RelL2(s), avg_Horizon(s)};
end
disp(cell2table(table3_data, 'VariableNames', {'State','RMSE','NRMSE','RelL2_Stable','Horizon_s'}));

figure('Color','w', 'Name', '4.3 Prediction Horizon');
b_rec = Phi \ H(:, 1); H_recon = zeros(size(H,1), train_win);
for k = 1:train_win, H_recon(:, k) = real(Phi * (exp(omega * (k-1) * dt) .* b_rec)); end
for s = 1:5
    subplot(5, 1, s);
    plot(time(n1:n2), X_raw(s, n1:n2), 'r', 'LineWidth', 1.5, 'DisplayName', 'Actual'); hold on;
    plot(time(n1:split_idx-1), H_recon(s,:), 'g--', 'DisplayName', 'Denoised');
    plot(time(split_idx+1:n2), X_dmd_pred(s,:), 'b--', 'LineWidth', 1.5, 'DisplayName', 'Forecast');
    fail_time = time(split_idx) + iter_Horizon(end, s);
    line([fail_time fail_time], ylim, 'Color', 'm', 'LineStyle', ':', 'LineWidth', 2, 'DisplayName', 'Horizon');
    ylabel(states(s).name); grid on;
    if s==1, title('4.3 Short-Term Prediction Performance'); legend('show','Location','best'); end
end