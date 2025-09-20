%% gpr_data_export.m
% Data → clean → GPR (fallback: linear) → export CSV for Origin figures
% Inputs : Data.xlsx (Sheet1)
% Outputs: Fig1_* ... Fig6_* CSV + Summary_Statistics.csv

clear; clc; rng(42);

%% -------- Config --------
inFile     = 'Data.xlsx';
inSheet    = 'Sheet1';
trainRatio = 0.8;
knnK       = 5;          % for simple 4D local avg on surface grid
maxTimePts = 500;        % for time series sample
maxGroupTs = 200;        % for grouped power time series
maWin      = 20;         % moving average window
stabWin    = 10;         % power stability window
alphaW     = 0.6;        % combined score weights
betaW      = 0.4;

%% -------- Load & column resolve --------
T = readtable(inFile,'Sheet',inSheet,'VariableNamingRule','preserve');
cols = T.Properties.VariableNames;

battery_set  = pick_var(T, {'Battery set','Battery_set','BatterySet','Battery'});
load_w       = pick_var(T, {'Load','Load_W'});
speed_rpm    = pick_var(T, find_name(cols,'Speed'));
pressure_n   = pick_var(T, find_name(cols,'Pressure'));
human_power  = pick_var(T, find_name(cols,'Human'));
gen_power    = pick_var(T, find_name(cols,'Gen'));
total_power  = pick_opt(T, find_name(cols,'Total'), human_power + gen_power);

%% -------- Parse battery / load to numbers --------
battery_values = parse_level(battery_set, [12 24 36 48]);   % regex-friendly
load_values    = parse_level(load_w,    [10 30 50 70]);     % adjust if需要

%% -------- Basic checks & ranges --------
efficiency = safe_div(gen_power, human_power);
efficiency = min(max(efficiency,0),1);

valid = (battery_values>0) & (load_values>0) & ...
        (speed_rpm>0) & (pressure_n>0) & (human_power>0) & ...
        (efficiency>0 & efficiency<1);

if nnz(valid) < 10
    error('有效数据太少：%d 条。请检查数据源/列名。', nnz(valid));
end

% restrict to valid
speed_rpm    = speed_rpm(valid);
pressure_n   = pressure_n(valid);
battery_vals = battery_values(valid);
load_vals    = load_values(valid);
human_pw     = human_power(valid);
gen_pw       = gen_power(valid);
eff          = efficiency(valid);

% normalize for model
[speed_z, mu_s, sig_s]   = zscore(speed_rpm);
[press_z, mu_p, sig_p]   = zscore(pressure_n);
[battery_z, mu_b, sig_b] = zscore(battery_vals);
[load_z, mu_l, sig_l]    = zscore(load_vals);

X = [speed_z, press_z, battery_z, load_z];
y = eff;

%% -------- Train / test split --------
N = size(X,1);
nTrain = floor(N*trainRatio);
idx = randperm(N);
itr = idx(1:nTrain); ite = idx(nTrain+1:end);

Xtr = X(itr,:); ytr = y(itr);
Xte = X(ite,:); yte = y(ite);

%% -------- GPR (fallback to linear) --------
try
    mdl = fitrgp(Xtr, ytr, 'KernelFunction','squaredexponential', 'Standardize',true);
    [yp,~,yci] = predict(mdl, Xte);
    rmse = sqrt(mean((yte-yp).^2));
    r2   = 1 - sum((yte-yp).^2)/sum((yte-mean(yte)).^2);
    modelTag = 'GPR-SE';
catch ME
    warning('GPR失败：%s。改用线性回归。', ME.message);
    mdl = fitlm(Xtr, ytr);
    yp  = predict(mdl, Xte);
    rmse = sqrt(mean((yte-yp).^2));
    r2   = mdl.Rsquared.Ordinary;
    se   = sqrt(mean((yte-yp).^2));                  % 粗略 CI
    yci  = [yp - 2*se, yp + 2*se];
    modelTag = 'Linear';
end

%% -------- Exports: Fig1 --------
% Fig1_1 Scatter (actual vs pred + CI)
writetable(array2table([yte*100, yp*100, yci(:,1)*100, yci(:,2)*100], ...
    'VariableNames',{'Actual_%','Pred_%','CI_L_%','CI_U_%'}), ...
    'Fig1_1_Prediction_Scatter.csv');

% Fig1_2 Heatmap (battery × load → mean efficiency)
bat_cfg = unique(battery_vals(battery_vals>0));
load_cfg= unique(load_vals(load_vals>0));
eff_mat = nan(numel(bat_cfg), numel(load_cfg));
for i=1:numel(bat_cfg)
    for j=1:numel(load_cfg)
        m = (battery_vals==bat_cfg(i)) & (load_vals==load_cfg(j));
        if any(m), eff_mat(i,j) = mean(eff(m))*100; end
    end
end
writematrix(eff_mat, 'Fig1_2_Efficiency_Heatmap.csv');

% Fig1_3 Residuals
writematrix((yte-yp)*100, 'Fig1_3_Residuals.csv');

% Fig1_4 QQ data
res = (yte-yp)*100;
res = res(isfinite(res));
res_s = sort(res);
theo  = norminv((1:numel(res))/(numel(res)+1), 0, std(res));
writetable(array2table([res_s(:), theo(:)], ...
    'VariableNames',{'Sample_Quantiles','Theoretical_Quantiles'}), ...
    'Fig1_4_QQ_Plot.csv');

%% -------- Exports: Fig2 (3D surface & contour by fixed battery/load) --------
% choose mid configs (fallback to first)
bi = max(1, ceil(numel(bat_cfg)/2));
li = max(1, ceil(numel(load_cfg)/2));
bat0 = bat_cfg(bi); load0 = load_cfg(li);

s_grid = linspace(min(speed_rpm),   max(speed_rpm),   20);
p_grid = linspace(min(pressure_n),  max(pressure_n),  20);
[Sg,Pg] = meshgrid(s_grid, p_grid);

% simple kNN avg in 4D z-space (speed, pressure, battery=bat0, load=load0)
Xz   = X;                                              % (N × 4)
Xq   = [(Sg(:)-mu_s)/sig_s, (Pg(:)-mu_p)/sig_p, ...
        (bat0-mu_b)/sig_b * ones(numel(Sg),1), ...
        (load0-mu_l)/sig_l* ones(numel(Sg),1)];
D    = pdist2(Xq, Xz);                                 % (M × N)
[~,I]= mink(D, knnK, 2);
Ef   = mean(y(I), 2)*100;
Ef   = reshape(Ef, size(Sg));

writematrix(Ef,  'Fig2_1_3D_Surface.csv');
writematrix(Sg,  'Fig2_1_Speed_Grid.csv');
writematrix(Pg,  'Fig2_1_Pressure_Grid.csv');
% Fig2_2：等高线直接用同一组文件

%% -------- Exports: Fig3 (importance / corr / optimal per config) --------
% 简单"重要性"：|corr|
imp = [abs(corr(speed_z,y,'Rows','complete')), ...
       abs(corr(press_z,y,'Rows','complete')), ...
       abs(corr(battery_z,y,'Rows','complete')), ...
       abs(corr(load_z,y,'Rows','complete'))];
imp = 100*imp/sum(imp);
writetable(array2table(imp,'VariableNames',{'Speed','Pressure','Battery','Load'}), ...
           'Fig3_1_Feature_Importance.csv');

Cvars = [speed_rpm, pressure_n, battery_vals, load_vals, y*100, human_pw];
Cmat  = corr(Cvars, 'Rows','pairwise');
writematrix(Cmat, 'Fig3_2_Correlation_Matrix.csv');

opt_rows = [];
for i=1:numel(bat_cfg)
    for j=1:numel(load_cfg)
        m = (battery_vals==bat_cfg(i)) & (load_vals==load_cfg(j));
        if any(m)
            [mx, ix] = max(y(m));
            idxAll = find(m);
            k = idxAll(ix);
            opt_rows = [opt_rows; bat_cfg(i), load_cfg(j), ...
                        speed_rpm(k), pressure_n(k), mx*100]; %#ok<AGROW>
        end
    end
end
if ~isempty(opt_rows)
    writetable(array2table(opt_rows, ...
        'VariableNames',{'Battery_V','Load_W','Optimal_Speed_RPM','Optimal_Pressure_N','Max_Eff_%'}), ...
        'Fig3_3_Optimal_Parameters.csv');
end

%% -------- Exports: Fig4 (distributions) --------
% box by load
[max_n, ~] = max_grp_len(load_vals, load_cfg);
M = nan(max_n, numel(load_cfg));
for j=1:numel(load_cfg)
    v = y(load_vals==load_cfg(j))*100;
    M(1:numel(v), j) = v;
end
writematrix(M,'Fig4_1_Load_Efficiency_Box.csv');

% power by battery
[max_n, ~] = max_grp_len(battery_vals, bat_cfg);
M = nan(max_n, numel(bat_cfg));
for i=1:numel(bat_cfg)
    v = human_pw(battery_vals==bat_cfg(i));
    M(1:numel(v), i) = v;
end
writematrix(M,'Fig4_2_Battery_Power_Violin.csv');

% speed / pressure hist inputs
writematrix(speed_rpm,  'Fig4_3_Speed_Distribution.csv');
writematrix(pressure_n, 'Fig4_4_Pressure_Distribution.csv');

%% -------- Exports: Fig5 (time series) --------
Tlen   = min(maxTimePts, numel(y));
ts     = (1:Tlen).';
eff_ts = y(1:Tlen)*100;
ma_ts  = movmean(eff_ts, maWin);
writetable(table(ts, eff_ts, ma_ts, 'VariableNames',{'Time_Index','Actual_%','Moving_Avg_%'}), ...
           'Fig5_1_Efficiency_TimeSeries.csv');

% grouped power by battery (first maxGroupTs points per group)
G = nan(maxGroupTs, 1+numel(bat_cfg));
G(:,1) = (1:maxGroupTs).';
for i=1:numel(bat_cfg)
    idx = find(battery_vals==bat_cfg(i));
    if ~isempty(idx)
        idx = idx(1:min(maxGroupTs, numel(idx)));
        G(1:numel(idx), i+1) = human_pw(idx);
    end
end
writematrix(G,'Fig5_2_Power_TimeSeries_Grouped.csv');

%% -------- Exports: Fig6 (combined score bubble) --------
stab = zeros(size(human_pw));
for t=stabWin+1:numel(human_pw)
    stab(t) = 1./(1+std(human_pw(t-stabWin:t)));
end
if max(stab)>0, stab = stab./max(stab); end
comb = alphaW*y + betaW*stab;

S = min(200, numel(y));
ix = randperm(numel(y), S);
tbl6 = table(y(ix)*100, stab(ix)*100, comb(ix)*100, battery_vals(ix), load_vals(ix), ...
    'VariableNames',{'Efficiency_%','Fitness_%','Combined_%','Battery_V','Load_W'});
writetable(tbl6,'Fig6_Bubble_Combined_Score.csv');

%% -------- Summary CSV --------
summary_stats = {
    'Model',              modelTag, '';
    'RMSE',               sprintf('%.4f', rmse), '';
    'R2',                 sprintf('%.4f', r2),   '';
    '', '', '';
    'Max Efficiency (%)', sprintf('%.2f', max(y)*100), '';
    'Mean Efficiency (%)',sprintf('%.2f', mean(y)*100), '';
    '', '', '';
    'Importance Speed (%)',   sprintf('%.2f', imp(1)), '';
    'Importance Pressure (%)',sprintf('%.2f', imp(2)), '';
    'Importance Battery (%)', sprintf('%.2f', imp(3)), '';
    'Importance Load (%)',    sprintf('%.2f', imp(4)), '';
    '', '', '';
    'N total',  sprintf('%d', height(T)), '';
    'N valid',  sprintf('%d', numel(y)), '';
    'Train',    sprintf('%d', size(Xtr,1)), '';
    'Test',     sprintf('%d', size(Xte,1)), '';
};
writetable(cell2table(summary_stats, 'VariableNames',{'Metric','Value','Note'}), ...
           'Summary_Statistics.csv');

disp('Done. CSV files generated.');

%% ===== helpers =====
function v = pick_var(T, candidates)
    if iscell(candidates)
        for k=1:numel(candidates)
            if ismember(candidates{k}, T.Properties.VariableNames)
                v = T.(candidates{k}); return;
            end
        end
        error('缺少列：%s', strjoin(candidates, ', '));
    else
        v = T.(candidates{1});
    end
end

function name = find_name(cols, key)
    ix = find(contains(cols, key, 'IgnoreCase', true), 1);
    if isempty(ix), error('找不到列关键字：%s', key); end
    name = cols{ix};
end

function v = pick_opt(T, name, fallback)
    if isempty(name), v = fallback; return; end
    if ismember(name, T.Properties.VariableNames)
        v = T.(name);
    else
        v = fallback;
    end
end

function out = parse_level(x, whitelist)
    % convert string like "48V" or "Load=70W" → numeric (48 / 70)
    out = zeros(size(x));
    if isnumeric(x), out = x; return; end
    xs = string(x);
    for i=1:numel(xs)
        tok = regexp(xs(i), '(-?\d+(\.\d+)?)', 'tokens', 'once');
        if ~isempty(tok)
            val = str2double(tok{1});
            if nargin>1 && ~isempty(whitelist)
                % snap to nearest whitelist value if within 10 units
                [d, j] = min(abs(whitelist - val));
                if d <= 10, val = whitelist(j); end
            end
            out(i) = val;
        else
            out(i) = 0;
        end
    end
end

function z = safe_div(a,b)
    b = max(b, eps);
    z = a./b;
    z(~isfinite(z)) = 0;
end

function [mx, which] = max_grp_len(v, levels)
    mx = 0; which = 1;
    for i=1:numel(levels)
        n = nnz(v==levels(i));
        if n>mx, mx=n; which=i; end
    end
end
