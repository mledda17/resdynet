%% generate_lti_rlc_dataset.m
% Dataset I/O per un sistema LTI fisico: circuito RLC serie
% Stato: x = [i; vC] (A: corrente nell'induttore, V: tensione sul condensatore)
% Ingresso: u = vin (V)
% Uscita:   y = vC  (V)
%
% Train/Val: stessi parametri nominali, ingressi PE (multisine + PRBS) con seed diversi
% Test: parametri diversi (shift) + ingresso PE diverso (banda/frequenze diverse)
%
% Salva in .mat una struct dataset con campi:
%   dataset.train(k).u, dataset.train(k).y, dataset.train(k).t
%   dataset.val(k).u,   dataset.val(k).y,   dataset.val(k).t
%   dataset.test(k).u,  dataset.test(k).y,  dataset.test(k).t
%
% Nota: usa discretizzazione (c2d) e simulazione iterativa per avere sequenze pulite.

clear; clc;

rng(1); % riproducibilità globale

%% =========================
%  Config generale
%  =========================
cfg.fs   = 200;          % Hz (sampling)
cfg.Ts   = 1/cfg.fs;     % s
cfg.Tend = 20;           % s durata di ciascuna sequenza
cfg.N    = round(cfg.Tend/cfg.Ts) + 1;
cfg.t    = (0:cfg.N-1)' * cfg.Ts;

% Numero di esperimenti per split
cfg.n_train = 40;
cfg.n_val   = 10;
cfg.n_test  = 15;

% Noise (misura) e disturbo processo (opzionale)
cfg.sigma_y = 5e-3;      % std rumore misura su y (V)
cfg.sigma_w = 0.0;       % std rumore processo (su stato) (mettilo >0 se vuoi)

% Saturazione ingresso (realistica)
cfg.u_max = 5.0;         % Volt
cfg.u_min = -5.0;

% Normalizzazione (salvata nel dataset)
cfg.do_normalize = true;

%% =========================
%  Modello fisico: RLC serie
%  =========================
% Dinamica continua:
%   di/dt   = (1/L)*(u - R*i - vC)
%   dvC/dt  = (1/C)*i
%
% => xdot = A x + B u, y = C x + D u
% Parametri nominali (train/val)
par_nom.R = 2.0;          % Ohm
par_nom.L = 50e-3;        % H
par_nom.C = 200e-6;       % F

sysd_nom = build_rlc_discrete(par_nom, cfg.Ts);

% Parametri test (diversi): variazione componenti (simula mismatch / altro circuito)
par_test.R = 2.5;         % Ohm (+25%)
par_test.L = 60e-3;       % H  (+20%)
par_test.C = 160e-6;      % F  (-20%)

sysd_test = build_rlc_discrete(par_test, cfg.Ts);

%% =========================
%  Generazione dataset
%  =========================
dataset = struct();
dataset.cfg = cfg;
dataset.par_nom  = par_nom;
dataset.par_test = par_test;

% Train
dataset.train = repmat(struct('t',[],'u',[],'y',[],'meta',[]), cfg.n_train, 1);
for k = 1:cfg.n_train
    seed = 1000 + k;
    [u, metaU] = generate_pe_input(cfg, seed, "train");
    y = simulate_lti(sysd_nom, u, cfg, seed);
    dataset.train(k).t = cfg.t;
    dataset.train(k).u = u;
    dataset.train(k).y = y;
    dataset.train(k).meta = metaU;
end

% Val
dataset.val = repmat(struct('t',[],'u',[],'y',[],'meta',[]), cfg.n_val, 1);
for k = 1:cfg.n_val
    seed = 2000 + k;
    [u, metaU] = generate_pe_input(cfg, seed, "val");
    y = simulate_lti(sysd_nom, u, cfg, seed);
    dataset.val(k).t = cfg.t;
    dataset.val(k).u = u;
    dataset.val(k).y = y;
    dataset.val(k).meta = metaU;
end

% Test (diverso: parametri diversi + ingresso diverso come banda/PRBS clock)
dataset.test = repmat(struct('t',[],'u',[],'y',[],'meta',[]), cfg.n_test, 1);
for k = 1:cfg.n_test
    seed = 3000 + k;
    [u, metaU] = generate_pe_input(cfg, seed, "test"); % differente per split
    y = simulate_lti(sysd_test, u, cfg, seed);
    dataset.test(k).t = cfg.t;
    dataset.test(k).u = u;
    dataset.test(k).y = y;
    dataset.test(k).meta = metaU;
end

%% =========================
%  (Opzionale) Normalizzazione usando solo TRAIN
%  =========================
if cfg.do_normalize
    % concatena train
    Utr = vertcat(dataset.train.u);
    Ytr = vertcat(dataset.train.y);

    mu_u = mean(Utr, 1);
    sd_u = std(Utr, 0, 1) + 1e-12;

    mu_y = mean(Ytr, 1);
    sd_y = std(Ytr, 0, 1) + 1e-12;

    dataset.norm.mu_u = mu_u;
    dataset.norm.sd_u = sd_u;
    dataset.norm.mu_y = mu_y;
    dataset.norm.sd_y = sd_y;

    % applica a tutti gli split (salva anche versioni normalizzate)
    dataset = apply_normalization(dataset);
end

%% =========================
%  Quick check / plot
%  =========================
figure; 
subplot(2,1,1);
plot(dataset.train(1).t, dataset.train(1).u, 'LineWidth', 1.0); grid on;
ylabel('u = v_{in} [V]');
title('Esempio TRAIN: ingresso persistently exciting');

subplot(2,1,2);
plot(dataset.train(1).t, dataset.train(1).y, 'LineWidth', 1.0); grid on;
ylabel('y = v_C [V]'); xlabel('t [s]');
title('Uscita (tensione sul condensatore)');

%% =========================
%  Salvataggio
%  =========================
out_file = fullfile(pwd, 'dataset_rlc_lti.mat');
save(out_file, 'dataset', '-v7');
fprintf('Salvato: %s\n', out_file);

%% ========================================================================
%  FUNZIONI LOCALI
%  ========================================================================

function sysd = build_rlc_discrete(par, Ts)
    R = par.R; L = par.L; C = par.C;

    A = [-(R/L)   -(1/L);
          (1/C)     0   ];
    B = [1/L; 0];
    Cmat = [0 1];  % y = vC
    D = 0;

    sysc = ss(A, B, Cmat, D);
    sysd = c2d(sysc, Ts, 'zoh');
end

function [u, meta] = generate_pe_input(cfg, seed, split)
    % Ingresso PE: multisine random-phase + PRBS, con parametri diversi per split.
    rng(seed);

    t = cfg.t;
    N = cfg.N;
    Ts = cfg.Ts;

    % --- multisine ---
    % Scelta banda e numero armoniche (diversa nel test)
    switch split
        case {"train","val"}
            fmin = 0.2;   % Hz
            fmax = 15.0;  % Hz
            K = 25;       % numero sinusoidi
            Ams = 1.5;    % ampiezza complessiva
        case "test"
            fmin = 0.5;   % Hz (banda diversa)
            fmax = 25.0;  % Hz
            K = 35;
            Ams = 1.2;
        otherwise
            error("split non riconosciuto");
    end

    % frequenze random nella banda (evita aliasing, <= fs/2)
    f = sort(fmin + (fmax - fmin) * rand(K,1));
    phi = 2*pi*rand(K,1);

    u_ms = zeros(N,1);
    for k = 1:K
        u_ms = u_ms + sin(2*pi*f(k)*t + phi(k));
    end
    u_ms = Ams * u_ms / max(abs(u_ms) + 1e-12);

    % --- PRBS ---
    % PRBS utile per eccitare dinamiche: clock diverso nel test
    switch split
        case {"train","val"}
            prbs_amp = 0.8;
            clock = 8; % samples per bit
        case "test"
            prbs_amp = 0.8;
            clock = 5; % più rapido -> diversa densità spettrale
    end

    nbits = ceil(N/clock);
    bits = randi([0 1], nbits, 1);
    prbs = repelem(2*bits-1, clock);
    prbs = prbs(1:N);
    u_prbs = prbs_amp * prbs;

    % mix + piccolo dithering
    u = 0.7*u_ms + 0.3*u_prbs + 0.05*randn(N,1);

    % saturazione
    u = min(max(u, cfg.u_min), cfg.u_max);

    meta = struct();
    meta.split = split;
    meta.seed = seed;
    meta.type = "multisine+prbs";
    meta.fmin = fmin;
    meta.fmax = fmax;
    meta.K = K;
    meta.prbs_clock = clock;
    meta.Ts = Ts;
end

function y = simulate_lti(sysd, u, cfg, seed)
    % Simulazione discreta x_{k+1} = Ad x_k + Bd u_k + w_k, y_k = Cd x_k + v_k
    rng(seed + 999);

    Ad = sysd.A; Bd = sysd.B;
    Cd = sysd.C; Dd = sysd.D;

    nx = size(Ad,1);
    N  = size(u,1);

    x = zeros(nx,1);  % condizione iniziale (a riposo)
    y = zeros(N,1);

    for k = 1:N
        v = cfg.sigma_y * randn(1,1);
        y(k) = Cd*x + Dd*u(k) + v;

        w = cfg.sigma_w * randn(nx,1);
        x = Ad*x + Bd*u(k) + w;
    end
end

function dataset = apply_normalization(dataset)
    mu_u = dataset.norm.mu_u;
    sd_u = dataset.norm.sd_u;
    mu_y = dataset.norm.mu_y;
    sd_y = dataset.norm.sd_y;

    % helper anonima
    normu = @(u) (u - mu_u) ./ sd_u;
    normy = @(y) (y - mu_y) ./ sd_y;

    for k = 1:numel(dataset.train)
        dataset.train(k).u_norm = normu(dataset.train(k).u);
        dataset.train(k).y_norm = normy(dataset.train(k).y);
    end
    for k = 1:numel(dataset.val)
        dataset.val(k).u_norm = normu(dataset.val(k).u);
        dataset.val(k).y_norm = normy(dataset.val(k).y);
    end
    for k = 1:numel(dataset.test)
        dataset.test(k).u_norm = normu(dataset.test(k).u);
        dataset.test(k).y_norm = normy(dataset.test(k).y);
    end
end
