%% generate_rlc_lti_dataset.m
clear; clc; close all;
rng(42);

%% Continuous-time RLC parameters
R = 2.0;        % Ohm
L = 0.5;        % Henry
C = 0.25;       % Farad

Ts = 0.05;      % Sampling time [s]

%% Continuous-time model
% State: x = [i_L; v_C]
% Input: u = voltage source
% Output: y = v_C

Ac = [-R/L   -1/L;
       1/C    0 ];

Bc = [1/L;
      0];

Cc = [0 1];
Dc = 0;

nx = size(Ac,1);
nu = size(Bc,2);
ny = size(Cc,1);

%% Discretization with zero-order hold
sysc = ss(Ac, Bc, Cc, Dc);
sysd = c2d(sysc, Ts, 'zoh');

A = sysd.A;
B = sysd.B;
Cmat = sysd.C;
D = sysd.D;

disp("Discrete-time eigenvalues:");
disp(eig(A));

%% Dataset lengths
N_train = 10000;
N_val   = 3000;
N_test  = 5000;
N_total = N_train + N_val + N_test;

%% Persistently exciting input
t = (0:N_total-1)' * Ts;

% Multisine component
n_freq = 25;
freqs = linspace(0.05, 4.0, n_freq);  % Hz
phases = 2*pi*rand(n_freq,1);

u_multisine = zeros(N_total,1);
for j = 1:n_freq
    u_multisine = u_multisine + sin(2*pi*freqs(j)*t + phases(j));
end
u_multisine = u_multisine / max(abs(u_multisine));

% PRBS-like component
switch_prob = 0.03;
u_prbs = zeros(N_total,1);
u_prbs(1) = sign(randn);

for k = 2:N_total
    if rand < switch_prob
        u_prbs(k) = -u_prbs(k-1);
    else
        u_prbs(k) = u_prbs(k-1);
    end
end

% Final input voltage
u = 1.0*u_multisine + 0.5*u_prbs + 0.05*randn(N_total,1);

% Input saturation
u_max = 2.0;
u = max(min(u, u_max), -u_max);

%% Simulate discrete-time system
x = zeros(N_total, nx);
y = zeros(N_total, ny);

x(1,:) = [0.0, 0.0];

for k = 1:N_total-1
    y(k,:) = Cmat*x(k,:)' + D*u(k,:)';
    x(k+1,:) = A*x(k,:)' + B*u(k,:)';
end

y(N_total,:) = Cmat*x(N_total,:)' + D*u(N_total,:)';

%% Add measurement noise only on output
add_noise = true;
noise_std = 0.01;

y_clean = y;

if add_noise
    y = y_clean + noise_std*randn(size(y_clean));
end

%% Split train / validation / test
idx_train = 1:N_train;
idx_val   = N_train+1:N_train+N_val;
idx_test  = N_train+N_val+1:N_total;

train.u = u(idx_train,:);
train.x = x(idx_train,:);
train.y = y(idx_train,:);
train.y_clean = y_clean(idx_train,:);

val.u = u(idx_val,:);
val.x = x(idx_val,:);
val.y = y(idx_val,:);
val.y_clean = y_clean(idx_val,:);

test.u = u(idx_test,:);
test.x = x(idx_test,:);
test.y = y(idx_test,:);
test.y_clean = y_clean(idx_test,:);

%% Save dataset
dataset.R = R;
dataset.L = L;
dataset.C = C;
dataset.Ts = Ts;

dataset.Ac = Ac;
dataset.Bc = Bc;
dataset.Cc = Cc;
dataset.Dc = Dc;

dataset.A = A;
dataset.B = B;
dataset.Cmat = Cmat;
dataset.D = D;

dataset.train = train;
dataset.val = val;
dataset.test = test;

save("rlc_lti_dataset.mat", "dataset");

%% Plot full dataset
figure;

subplot(3,1,1);
plot(t, u, 'LineWidth', 1);
grid on;
ylabel("u_k [V]");
title("Input voltage");

subplot(3,1,2);
plot(t, x(:,1), 'LineWidth', 1); hold on;
plot(t, x(:,2), 'LineWidth', 1);
grid on;
ylabel("x_k");
legend("i_L [A]", "v_C [V]");

subplot(3,1,3);
plot(t, y, 'LineWidth', 1); hold on;
plot(t, y_clean, '--', 'LineWidth', 1);
grid on;
ylabel("y_k [V]");
xlabel("time [s]");
legend("measured y", "clean y");
title("Output voltage");

%% Persistency check through Hankel matrix
Lh = 20;
H = build_hankel(u, Lh);
rank_H = rank(H, 1e-8);

fprintf("Hankel rank for input with L = %d: %d / %d\n", Lh, rank_H, Lh);

if rank_H == Lh
    fprintf("Input is numerically persistently exciting of order %d.\n", Lh);
else
    warning("Input may not be persistently exciting of order %d.", Lh);
end

%% Local function
function H = build_hankel(signal, Lh)
    N = size(signal,1);
    cols = N - Lh + 1;
    H = zeros(Lh, cols);

    for i = 1:Lh
        H(i,:) = signal(i:i+cols-1)';
    end
end