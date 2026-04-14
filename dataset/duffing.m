% ===== generate_duffing_dataset.m =====
clear; clc; rng(42);

% Duffing params
Ts = 0.01; delta = 0.2; alpha = 1.0; beta = 1.0; gamma = 1.0;

% lengths
N_train = 45000;
N_val   = 5000;
N_test  = 10000;
N_total = N_train + N_val + N_test;

% Input: white, i.i.d. uniform in (-2, 2)
u = -2 + 4*rand(N_total,1);   % U(-2,2)

% Simulate Duffing (CLEAN)
x = zeros(N_total,2);
y_clean = zeros(N_total,1);
for k = 1:N_total-1
    x1 = x(k,1); x2 = x(k,2);
    uk = u(k);

    x(k+1,1) = x1 + Ts * x2;
    x(k+1,2) = x2 + Ts * ( -delta*x2 - alpha*x1 - beta*(x1^3) + gamma*uk );
    y_clean(k) = x1 + 0.05*(x1^2);
end
y_clean(end) = x(end,1) + 0.05*(x(end,1)^2);

% ----------------------------
% Add measurement noise ONLY to train + val
% ----------------------------
% SNR_dB = 40;

% signal power on the portion we will noise (train+val)
% y_tv = y_clean(1 : N_train + N_val);
% Ps = mean(y_tv.^2);
% Pn = Ps / (10^(SNR_dB/10));

%v_trainval = sqrt(Pn) * randn(N_train + N_val, 1);

% build final y
y = y_clean;  % start from clean
y(1 : N_train + N_val) = y(1 : N_train + N_val); %+ v_trainval;
% y(N_train+N_val+1:end) stays clean for test

% Split
u_train = u(1:N_train);
y_train = y(1:N_train);

u_val   = u(N_train+1:N_train+N_val);
y_val   = y(N_train+1:N_train+N_val);

u_test  = u(N_train+N_val+1:end);
y_test  = y_clean(N_train+N_val+1:end);   % explicitly store CLEAN test

% Save
save('dataset_duffing.mat', ...
     'u_train','y_train', ...
     'u_val','y_val', ...
     'u_test','y_test', ...
     'Ts');

disp('Saved dataset_duffing.mat (noise only on train+val, clean test)');
