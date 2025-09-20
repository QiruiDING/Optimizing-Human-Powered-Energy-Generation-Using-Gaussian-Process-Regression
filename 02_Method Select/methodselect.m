%% human_power_pipeline.m
% Human-powered energy: data load → clean → model select → search → plots → export
% Input:  Data.xlsx / Sheet1
% Output: figures (pdf/eps/png/fig) + stdout summary

function human_power_pipeline()
    clc; close all; rng(42);

    %% ---------- Config ----------
    inFile  = 'Data.xlsx';
    inSheet = 'Sheet1';
    comfortQuant = [0.20 0.80];   % speed/pressure comfort bounds (quantiles)
    resGrid = 60;                 % grid resolution for search
    kfold   = 5;                  % CV folds
    iqrK    = 3;                  % IQR multiplier (outlier gate)
    exportBase = 'HumanPower_Optimization';

    %% ---------- Load & tidy ----------
    T = load_table(inFile, inSheet);
    T = standardize_columns(T);
    check_required(T, {'Battery_set','Load','Timestamp','Speed_RPM','Pressure_N','Human_Power_W','Total_Power_W'});

    % type & basic filter
    if ~isdatetime(T.Timestamp), T.Timestamp = datetime(T.Timestamp); end
    T = sortrows(T, {'Battery_set','Load','Timestamp'});
    T = T(T.Human_Power_W>0 & T.Speed_RPM>0 & T.Pressure_N>0, :);

    % efficiency
    Pin  = max(T.Human_Power_W, eps);
    Pout = max(T.Total_Power_W, 0);
    T.eta = Pout ./ Pin;

    % outlier by IQR on 3 vars
    keep = iqr_keep(T, {'Speed_RPM','Pressure_N','eta'}, iqrK);
    T = T(keep, :);

    % categorical
    T.Battery_set = make_categorical(T.Battery_set);
    T.Load        = make_categorical(T.Load);

    batLevels = categories(T.Battery_set);
    loadLevels = categories(T.Load);

    % model table
    Xnum = table(double(T.Speed_RPM), double(T.Pressure_N), ...
                 double(T.Battery_set), double(T.Load), ...
                 'VariableNames', {'Speed','Pressure','Battery','Load'});
    y = T.eta;

    %% ---------- Model compare ----------
    cvp = cvpartition(y,'KFold',kfold);
    evals = struct('name',{},'R2',{});

    % GAM (if toolbox present)
    [bestGam, r2Gam] = fit_gam(T, y, cvp);
    evals(end+1) = struct('name','GAM','R2',r2Gam); %#ok<SAGROW>

    % GPR
    [bestGpr, r2Gpr] = fit_gpr(Xnum, y, cvp);
    evals(end+1) = struct('name','GPR','R2',r2Gpr); %#ok<SAGROW>

    % LSBoost
    [bestEns, r2Ens] = fit_lsboost(Xnum, y, cvp);
    evals(end+1) = struct('name','LSBoost','R2',r2Ens); %#ok<SAGROW>

    % choose best
    [~,ix] = max([evals.R2]);
    bestName = evals(ix).name;
    switch bestName
        case 'GAM',     bestModel = bestGam; XisCat = true;
        case 'GPR',     bestModel = bestGpr; XisCat = false;
        case 'LSBoost', bestModel = bestEns; XisCat = false;
        otherwise, error('No valid model selected.');
    end

    %% ---------- Grid search with comfort bounds ----------
    pack = struct([]);
    optTable = table();
    row = 0;

    for bi = 1:numel(batLevels)
        for li = 1:numel(loadLevels)
            mask = T.Battery_set==batLevels{bi} & T.Load==loadLevels{li};
            if ~any(mask), continue; end

            sp = T.Speed_RPM(mask);
            pr = T.Pressure_N(mask);

            loS = quantile(sp,comfortQuant(1)); hiS = quantile(sp,comfortQuant(2));
            loP = quantile(pr,comfortQuant(1)); hiP = quantile(pr,comfortQuant(2));
            sGrid = linspace(loS,hiS,resGrid);
            pGrid = linspace(loP,hiP,resGrid);
            [S,P] = meshgrid(sGrid,pGrid);

            if XisCat
                Xg = table(S(:), P(:), ...
                       categorical(repmat(batLevels(bi),numel(S),1), batLevels), ...
                       categorical(repmat(loadLevels(li),numel(S),1), loadLevels), ...
                       'VariableNames', {'Speed','Pressure','Battery','Load'});
            else
                Xg = table(S(:), P(:), repmat(bi,numel(S),1), repmat(li,numel(S),1), ...
                       'VariableNames', {'Speed','Pressure','Battery','Load'});
            end

            etaPred = predict(bestModel, Xg);
            [etaMax, idx] = max(etaPred);
            sStar = S(idx); pStar = P(idx);

            row = row + 1;
            optTable(row,:) = table( string(batLevels{bi}), string(loadLevels{li}), ...
                loS, hiS, loP, hiP, sStar, pStar, etaMax, ...
                'VariableNames', {'Battery','Load','ComfortSpeedLo','ComfortSpeedHi', ...
                'ComfortPressLo','ComfortPressHi','SpeedStar','PressStar','EtaPredMax'});

            pack(bi,li).S=S; pack(bi,li).P=P; pack(bi,li).Eta=reshape(etaPred,size(S));
            pack(bi,li).sStar=sStar; pack(bi,li).pStar=pStar;
            pack(bi,li).sp=sp; pack(bi,li).pr=pr;
        end
    end

    %% ---------- Simple state labelling (GMM) ----------
    Z = zscore([double(T.Speed_RPM), double(T.Pressure_N)]);
    T.zSpeed = Z(:,1); T.zPressure = Z(:,2);

    if ismember('Testname', T.Properties.VariableNames)
        seqGroups = findgroups(T.Testname);
    else
        seqGroups = findgroups(T.Battery_set, T.Load);
    end

    [stateLbl, trans] = learn_states_gmm(Z, seqGroups);

    %% ---------- Pareto (discomfort vs. -eta) ----------
    [discomfort, etaRep, isPareto] = pareto_block(T);

    %% ---------- Plot ----------
    fig = make_figure(pack, batLevels, loadLevels, discomfort, etaRep, isPareto);
    [bestR2,besti] = max([evals.R2]); bestTag = sprintf('%s (R^2=%.3f)', evals(besti).name, bestR2);

    %% ---------- Export ----------
    export_all(fig, exportBase);
    fprintf('\n== Summary ==\n');
    fprintf('Best model: %s\n', bestTag);
    fprintf('Opt table rows: %d\n', height(optTable));
    fprintf('States: %d, transition rows=%d\n', max(stateLbl), size(trans,1));
end

%% ======================= helpers =======================

function T = load_table(file, sheet)
    if ~isfile(file)
        error('Input file not found: %s', file);
    end
    opts = detectImportOptions(file,'Sheet',sheet);
    opts.VariableNamingRule = 'preserve';
    T = readtable(file, opts);
end

function T = standardize_columns(T)
    if ismember('Battery set', T.Properties.VariableNames)
        T.Properties.VariableNames{'Battery set'} = 'Battery_set';
    end
    if ismember('Total_PowerW', T.Properties.VariableNames) && ...
       ~ismember('Total_Power_W', T.Properties.VariableNames)
        T.Properties.VariableNames{'Total_PowerW'} = 'Total_Power_W';
    end
end

function check_required(T, must)
    miss = setdiff(must, T.Properties.VariableNames);
    assert(isempty(miss), 'Missing columns: %s', strjoin(miss,','));
end

function c = make_categorical(c)
    if ischar(c) || isstring(c), c = categorical(cellstr(c)); end
    if ~iscategorical(c), c = categorical(c); end
end

function keep = iqr_keep(T, varnames, k)
    keep = true(height(T),1);
    for i=1:numel(varnames)
        x = T.(varnames{i});
        q1 = quantile(x,0.25); q3 = quantile(x,0.75); iq = q3-q1;
        lo = q1 - k*iq; hi = q3 + k*iq;
        keep = keep & x>=lo & x<=hi;
    end
end

function [mdl, r2] = fit_gam(T, y, cvp)
    mdl = []; r2 = NaN;
    if exist('fitrgam','file') ~= 2, return; end
    try
        Xgam = table(double(T.Speed_RPM), double(T.Pressure_N), T.Battery_set, T.Load, ...
            'VariableNames', {'Speed','Pressure','Battery','Load'});
        cv = fitrgam(Xgam, y, 'CategoricalPredictors',[3 4], ...
                     'CrossVal','on','CVPartition',cvp, 'Interactions','all','Verbose',0);
        yhat = kfoldPredict(cv);
        r2 = 1 - sse(y, yhat);
        mdl = cv.Trained{1};
    catch
        % NOTE: fall through with NaN r2
    end
end

function [mdl, r2] = fit_gpr(X, y, cvp)
    mdl = []; r2 = NaN;
    try
        cv = fitrgp(X, y, 'CategoricalPredictors',[3 4], ...
            'KernelFunction','ardsquaredexponential', 'Standardize',true, ...
            'CrossVal','on','CVPartition',cvp);
        yhat = kfoldPredict(cv);
        r2 = 1 - sse(y, yhat);
        mdl = cv.Trained{1};
    catch
    end
end

function [mdl, r2] = fit_lsboost(X, y, cvp)
    mdl = []; r2 = NaN;
    try
        cv = fitrensemble(X, y, 'CategoricalPredictors',[3 4], ...
            'Method','LSBoost','NumLearningCycles',300,'LearnRate',0.05, ...
            'CrossVal','on','CVPartition',cvp);
        yhat = kfoldPredict(cv);
        r2 = 1 - sse(y, yhat);
        mdl = cv.Trained{1};
    catch
    end
end

function v = sse(y, yhat)
    v = sum((y - yhat).^2)/sum((y-mean(y)).^2);
end

function [stateLbl, trans] = learn_states_gmm(Z, seqGroups)
    Kcands = 3:5;
    bestBIC = inf; bestG = [];
    for K = Kcands
        try
            g = fitgmdist(Z, K, 'RegularizationValue',1e-4, 'Replicates',3, ...
                          'Options',statset('MaxIter',500));
            if g.BIC < bestBIC, bestBIC = g.BIC; bestG = g; end
        catch
        end
    end
    if ~isempty(bestG)
        stateLbl = cluster(bestG, Z);
    else
        % fallback
        K = 3; stateLbl = kmeans(Z, K);
    end

    K = max(stateLbl);
    trans = zeros(K);
    ug = unique(seqGroups);
    for gi = 1:numel(ug)
        idx = find(seqGroups==ug(gi));
        s = stateLbl(idx);
        for t = 1:numel(s)-1
            trans(s(t), s(t+1)) = trans(s(t), s(t+1)) + 1;
        end
    end
    trans = bsxfun(@rdivide, trans, max(sum(trans,2),1));
end

function [discomfort, etaRep, isPareto] = pareto_block(T)
    % choose the Battery×Load with most samples as representative
    batLevels = categories(T.Battery_set); loadLevels = categories(T.Load);
    repB = []; repL = []; maxN = 0;
    for bi=1:numel(batLevels)
        for li=1:numel(loadLevels)
            n = nnz(T.Battery_set==batLevels{bi} & T.Load==loadLevels{li});
            if n>maxN, maxN=n; repB=batLevels{bi}; repL=loadLevels{li}; end
        end
    end
    if isempty(repB), repB=batLevels{1}; end
    if isempty(repL), repL=loadLevels{1}; end

    m = T.Battery_set==repB & T.Load==repL;
    if ~any(m), m = true(height(T),1); end

    sp = T.Speed_RPM(m); pr = T.Pressure_N(m);
    zs = (sp-mean(sp))/std(sp); zp = (pr-mean(pr))/std(pr);
    discomfort = zs.^2 + zp.^2;
    etaRep = T.eta(m);

    Y = [discomfort, -etaRep];
    isPareto = pareto_front(Y);
end

function fig = make_figure(pack, batLevels, loadLevels, discomfort, etaRep, isPareto)
    fig = figure('Units','centimeters','Position',[2 2 20 22],'Color','w');

    t = tiledlayout(fig,5,4,'Padding','compact','TileSpacing','compact');

    % (a) 4x4 contours
    for li = 1:min(4,numel(loadLevels))
        for bi = 1:min(4,numel(batLevels))
            idx = (li-1)*4 + bi;
            ax = nexttile(t, idx); hold(ax,'on'); box(ax,'on'); grid(ax,'on'); set(ax,'Layer','top');
            ok = (bi<=size(pack,1) && li<=size(pack,2) && isfield(pack(bi,li),'S') && ~isempty(pack(bi,li).S));
            if ~ok, axis(ax,'off'); continue; end

            S=pack(bi,li).S; P=pack(bi,li).P; Eta=pack(bi,li).Eta;
            scatter(ax, pack(bi,li).sp, pack(bi,li).pr, 8, [0.4 0.4 0.4], 'filled', 'MarkerFaceAlpha',0.35);
            contourf(ax, S, P, Eta*100, 12, 'LineStyle','none'); % percent
            plot(ax, pack(bi,li).sStar, pack(bi,li).pStar, 'p', 'MarkerSize',8, 'MarkerFaceColor',[1 0.8 0], 'MarkerEdgeColor',[0.6 0.3 0]);

            if bi==1, ylabel(ax,'Pressure (N)'); else, yticklabels(ax,[]); end
            if li==4, xlabel(ax,'Speed (RPM)'); else, xticklabels(ax,[]); end
            title(ax, sprintf('%s V, %s W', string(batLevels{bi}), string(loadLevels{li})), 'FontSize',9);
        end
    end

    % (b) placeholder for timeseries/state —此处视数据可改为你自己的叙事
    ax_b = nexttile(t,17,[1 2]); axis(ax_b,'off'); text(ax_b,0,0.5,'(b) State timeline: add if needed','FontSize',10);

    % (c) Pareto
    ax_c = nexttile(t,19,[1 2]); hold(ax_c,'on'); box(ax_c,'on'); grid(ax_c,'on');
    sc1 = scatter(ax_c, discomfort, etaRep, 20, [0.6 0.6 0.6], 'filled', 'MarkerFaceAlpha',0.6);
    if any(isPareto)
        sc2 = scatter(ax_c, discomfort(isPareto), etaRep(isPareto), 36, [0.85 0.2 0.1], 'filled');
        legend(ax_c,[sc1 sc2],{'All points','Pareto'},'Box','off','Location','northeast');
    else
        legend(ax_c,sc1,{'All points'},'Box','off','Location','northeast');
    end
    xlabel(ax_c,'Discomfort (z^2 sum)');
    ylabel(ax_c,'Efficiency \eta');
end

function export_all(fig, base)
    % vector
    try
        exportgraphics(fig, sprintf('%s.pdf', base), 'ContentType','vector', 'BackgroundColor','white');
        exportgraphics(fig, sprintf('%s.eps', base), 'ContentType','vector', 'BackgroundColor','white');
    catch ME
        warning('Vector export failed: %s', ME.message);
    end
    % raster
    try
        exportgraphics(fig, sprintf('%s.png', base), 'Resolution', 400, 'BackgroundColor','white');
    catch
        try
            print(fig, sprintf('%s.png', base), '-dpng', '-r400');
        catch ME
            warning('PNG export failed: %s', ME.message);
        end
    end
    savefig(fig, sprintf('%s.fig', base));
end

function isP = pareto_front(Y)
    n = size(Y,1);
    isP = true(n,1);
    for i=1:n
        % dominated if ∃ j: Y(j, :) ≤ Y(i, :) & Y(j, :) < Y(i, :)
        if any(all(bsxfun(@le, Y, Y(i,:)) & bsxfun(@lt, Y, Y(i,:)), 2))
            isP(i) = false;
        end
    end
end
