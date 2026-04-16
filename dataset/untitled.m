%% generate_linear_lti_dataset.m
clear; clc; close all;
rng(42);

%% System definition
A = [ 0.8   0.2;
     -0.1   0.9];

B = [0.1;
     0.05];

C = [1 0];

D = 0;

nx = size(A,1);
nu = size(B,2);
ny = size(C,1);

%% Dataset lengths
N_train = 10000;
N_val   = 3000;
N_test  = 5000;

N_total = N_train + N_val + N_test;

%% Input generation
% Rich input: sum of sinusoids + PRBS + small white noise.
% This is persistently exciting in practice for linear system identification.

Ts = 1.0;
t = (0:N_total-1)' * Ts;

n_freq = 20;
freqs = linspace(0.005, 0.25, n_freq);
phases = 2*pi*rand(n_freq,1);

u_multisine = zeros(N_total,1);
for i = 1:n_freq
    u_multisine = u_multisine + sin(2*pi*freqs(i)*t + phases(i));
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

% Final input
u = 0.7*u_multisine + 0.3*u_prbs + 0.05*randn(N_total,1);

% Saturate input
u = max(min(u, 1), -1);

%% Simulate system
x = zeros(N_total, nx);
y = zeros(N_total, ny);

x(1,:) = [0.5 -0.3];

for k = 1:N_total-1
    y(k,:) = C*x(k,:)' + D*u(k,:)';
    x(k+1,:) = A*x(k,:)' + B*u(k,:)';
end

y(N_total,:) = C*x(N_total,:)' + D*u(N_total,:)';

%% Optional output noise
add_noise = true;
noise_std = 0.01;

if add_noise
    y_clean = y;
    y = y + noise_std*randn(size(y));
else
    y_clean = y;
end

%% Split dataset
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

%% Save
dataset.A = A;
dataset.B = B;
dataset.C = C;
dataset.D = D;
dataset.Ts = Ts;
dataset.train = train;
dataset.val = val;
dataset.test = test;

save("linear_lti_dataset.mat", "dataset");

%% Plot
figure;
subplot(3,1,1);
plot(u);
grid on;
ylabel("u_k");
title("Input");

subplot(3,1,2);
plot(x);
grid on;
ylabel("x_k");
legend("x_1","x_2");

subplot(3,1,3);
plot(y);
grid on;
ylabel("y_k");
xlabel("k");
title("Output");

%% Basic persistency check via Hankel matrix rank
L = 20;                  % window length
H = build_hankel(u, L);  % L-block Hankel for SISO input
rank_H = rank(H, 1e-8);

fprintf("Hankel rank for input with L = %d: %d / %d\n", L, rank_H, L);

if rank_H == L
    fprintf("Input is numerically persistently exciting of order %d.\n", L);
else
    warning("Input may not be sufficiently persistently exciting of order %d.", L);
end

%% Local function
function H = build_hankel(signal, L)
    N = size(signal,1);
    cols = N - L + 1;
    H = zeros(L, cols);

    for i = 1:L
        H(i,:) = signal(i:i+cols-1)';
    end
end