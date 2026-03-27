function node = run_mrdmd_recursive(X, dt, level, max_level, rho_base)
    
    % Get dimensions
    [n_vars, n_time] = size(X);
    
    % Initialize node
    node.level = level;
    node.left = [];
    node.right = [];
    
    % --- Step 1: Standard DMD ---
    % Use a safe rank truncation (e.g., keep 99% energy)
    [Phi, Omega, lambda, b, X_dmd_full] = standard_dmd_robust(X, dt);
    
    % --- Step 2: Identify Slow Modes ---
    % Scale rho: The deeper we go, the higher the frequency cutoff can be
    % or we keep it fixed relative to the window length.
    % Paper suggestion: rho inversely proportional to window length T.
    % T_level = n_time * dt;
    % rho = constant / T_level;
    
    % Simple geometric scaling:
    rho_current = rho_base * (2^level); 
    
    slow_idx = find(abs(imag(Omega)) <= rho_current);
    fast_idx = find(abs(imag(Omega)) > rho_current);
    
    % --- Step 3: Reconstruct Slow Component ---
    if ~isempty(slow_idx)
        % Reconstruct using ONLY slow modes
        % X_slow = Phi_slow * diag(exp...) * b_slow
        X_slow = reconstruct_modes(Phi, Omega, b, slow_idx, n_time, dt);
    else
        X_slow = zeros(n_vars, n_time);
    end
    
    node.reconstruction = real(X_slow);
    
    % --- Step 4: Subtract Slow to get Fast Component ---
    X_fast = X - node.reconstruction;
    
    % --- Step 5: Recursion (Split Data) ---
    if level < max_level
        mid_point = floor(n_time / 2);
        
        % Split the FAST component (Residual)
        X_left  = X_fast(:, 1:mid_point);
        X_right = X_fast(:, mid_point+1:end);
        
        node.left  = run_mrdmd_recursive(X_left, dt, level+1, max_level, rho_base);
        node.right = run_mrdmd_recursive(X_right, dt, level+1, max_level, rho_base);
    end
end

function X_rec = reconstruct_modes(Phi, Omega, b, idx, n_time, dt)
    t = (0:n_time-1) * dt;
    % Efficient computation
    % X = Phi * (b .* exp(omega*t))
    Modes = Phi(:, idx);
    Amps  = b(idx);
    Freqs = Omega(idx);
    
    TimeDyn = exp(Freqs * t) .* Amps;
    X_rec = Modes * TimeDyn;
end

function [Phi, Omega, lambda, b, X_dmd] = standard_dmd_robust(X, dt)
    % Robust DMD with hard rank check
    X1 = X(:, 1:end-1);
    X2 = X(:, 2:end);
    
    [U, S, V] = svd(X1, 'econ');
    
    % Rank truncation (Energy based)
    s_vals = diag(S);
    energy = cumsum(s_vals) / sum(s_vals);
    r = find(energy >= 0.999, 1); % Keep 99.9% of variance
    if isempty(r), r = size(S,1); end
    
    Ur = U(:, 1:r);
    Sr = S(1:r, 1:r);
    Vr = V(:, 1:r);
    
    Atilde = Ur' * X2 * Vr / Sr;
    [W, D] = eig(Atilde);
    lambda = diag(D);
    Omega = log(lambda) / dt;
    
    Phi = X2 * Vr / Sr * W;
    
    % Compute amplitudes b (Project x1 onto modes)
    b = Phi \ X1(:,1);
    
    % Full reconstruction for this step (internal check)
    X_dmd = zeros(size(X));
end