function draw()
% DRAW
% Build three figures (validation, mechanism, dynamics) from CSV files and
% export them to figs_ieee/ in PDF/SVG/PNG/EPS.
%
% Folder layout: put draw.m and all required Fig*.csv files in the same folder.

%% --- Setup ---
inDir  = fileparts(mfilename('fullpath'));
outDir = fullfile(inDir, 'figs_ieee');
if ~exist(outDir,'dir'), mkdir(outDir); end

S = styleIEEE();         % visual style in one place
applyRootStyle(S);       % light global defaults

% Simple toggles (put likely adjustments here)
USE_IQR_FILTER = true;   % outlier gating by IQR on (obs,pred)
USE_MAD_FILTER = true;   % outlier gating by MAD on (obs,pred)
IQR_K  = 3.0;            % IQR multiplier
MAD_TH = 3.5;            % MAD threshold
USE_RANSAC = true;       % robust fit line overlay

%% --- Read CSVs ---
files = { ...
 'Fig1_1_Prediction_Scatter.csv', ...
 'Fig1_2_Efficiency_Heatmap.csv', ...
 'Fig1_3_Residuals.csv', ...
 'Fig1_4_QQ_Plot.csv', ...
 'Fig2_1_3D_Surface.csv', ...
 'Fig2_1_Pressure_Grid.csv', ...
 'Fig2_1_Speed_Grid.csv', ...
 'Fig3_1_Feature_Importance.csv', ...
 'Fig3_2_Correlation_Matrix.csv', ...
 'Fig3_3_Optimal_Parameters.csv', ...
 'Summary_Statistics.csv', ...
 'Fig4_1_Load_Efficiency_Box.csv', ...
 'Fig4_2_Battery_Power_Violin.csv', ...
 'Fig4_3_Speed_Distribution.csv', ...
 'Fig4_4_Pressure_Distribution.csv', ...
 'Fig5_1_Efficiency_TimeSeries.csv', ...
 'Fig5_2_Power_TimeSeries_Grouped.csv', ...
 'Fig6_Bubble_Combined_Score.csv' ...
};

T = struct();
d = dir(fullfile(inDir,'*.csv'));
M = containers.Map('KeyType','char','ValueType','char');
for i=1:numel(d), M(lower(d(i).name)) = d(i).name; end
for i=1:numel(files)
    f = files{i}; key = lower(f);
    if isKey(M,key), pth = fullfile(inDir, M(key)); else, pth = fullfile(inDir, f); end
    Tbl = readCsvSmart(pth);
    if ~isempty(Tbl), T.(makeVar(f)) = Tbl; end
end

%% ================= Figure 1: Model Validation (2×2) =================
fig1 = newFig('double', S);
tiledlayout(fig1,2,2,'TileSpacing','compact','Padding','compact');

% (a) Observed vs Predicted
nexttile; hold on;
[obs_all, pred_all] = findPredObs(getOrEmpty(T,'Fig1_1_Prediction_Scatter'));
if isempty(obs_all)
    noData('(a) Observed vs Predicted', S);
else
    x = obs_all(:); y = pred_all(:);
    if USE_IQR_FILTER, [x,y] = iqrGate(x,y,IQR_K); end
    if USE_MAD_FILTER, [x,y] = madGate(x,y,MAD_TH); end

    % scatter
    safeScatter(x, y, S.ms*0.7, S.c(1,:), 0.65);

    % identity and fit
    dmin = min([x;y]); dmax = max([x;y]); pad = 0.03*(dmax-dmin+eps);
    xx = [dmin-pad, dmax+pad];
    plot(xx, xx, '--', 'Color', S.gray2, 'LineWidth', S.lwThin);

    if USE_RANSAC
        [xin, yin, inMask] = ransacLine(x, y, 0.02, 400);
        p = polyfit(xin, yin, 1);
        plot(xx, polyval(p,xx), '-', 'Color', S.c(2,:), 'LineWidth', S.lw);
        % optional: ellipse for inliers
        try, confEllipse(xin, yin, 0.95, S.c(6,:), S.lwThin); end
        [R2, RMSE, ~, ~] = metrics(xin,yin);
    else
        p = polyfit(x, y, 1);
        plot(xx, polyval(p,xx), '-', 'Color', S.c(2,:), 'LineWidth', S.lw);
        try, confEllipse(x, y, 0.95, S.c(6,:), S.lwThin); end
        [R2, RMSE, ~, ~] = metrics(x,y);
    end

    xlim(xx); ylim(xx);
    title('(a) Observed vs Predicted', S.ttl{:});
    xlabel('Observed', S.ax{:}); ylabel('Predicted', S.ax{:});
    set(gca,S.gca{:}); grid on;

    lgd = legend({'Data','Identity','Fit','95% CI'}, 'Location','northeast','FontSize',S.fsLeg); lgd.Box='off';
    annotationText = sprintf('R^2 = %.3f\nRMSE = %.3f', R2, RMSE);
    text(0.05,0.95,annotationText,'Units','normalized','VerticalAlignment','top',...
        'FontSize',S.fsTick,'FontName',S.font,'BackgroundColor',[1 1 1 0.85]);
end

% (b) Residuals vs Fitted
nexttile; hold on;
if isempty(obs_all)
    noData('(b) Residuals vs Fitted', S);
else
    fitted = pred_all(:); resid = obs_all(:) - pred_all(:);
    m = isfinite(fitted) & isfinite(resid);
    fitted = fitted(m); resid = resid(m);

    safeScatter(fitted, resid, S.ms*0.6, S.c(3,:), 0.55);
    yline(0,'-','Color',S.gray2,'LineWidth',S.lwThin);

    [fs,idx] = sort(fitted); rr = resid(idx);
    k = max(5, floor(numel(rr)/20));
    tr = movmedian(rr, k, 'omitnan');
    sw = max(5, floor(numel(rr)/25));
    sd = movstd(rr, sw, 'omitnan');
    plot(fs, tr,'-','Color',S.c(5,:),'LineWidth',S.lw);
    fill([fs;flipud(fs)], [tr-2*sd; flipud(tr+2*sd)], S.c(5,:), ...
        'FaceAlpha',0.10,'EdgeColor','none');

    title('(b) Residuals vs Fitted', S.ttl{:});
    xlabel('Fitted', S.ax{:}); ylabel('Residual', S.ax{:});
    set(gca,S.gca{:}); grid on;
    lgd = legend({'Residuals','Zero','Trend','95% band'}, 'Location','northeast','FontSize',S.fsLeg); lgd.Box='off';
end

% (c) Q-Q Plot
nexttile; hold on;
if isempty(obs_all)
    noData('(c) Q-Q Plot', S);
else
    resid = obs_all(:) - pred_all(:);
    resid = resid(isfinite(resid));
    resid = (resid - mean(resid,'omitnan'))/std(resid,1,'omitnan');
    [qqx, qqy] = qqPoints(resid);
    safeScatter(qqx, qqy, S.ms*0.6, S.c(4,:), 0.7);
    qmin=min([qqx;qqy]); qmax=max([qqx;qqy]);
    plot([qmin qmax],[qmin qmax],'--','Color',S.gray2,'LineWidth',S.lwThin);

    title('(c) Q-Q Plot', S.ttl{:});
    xlabel('Theoretical Quantiles', S.ax{:});
    ylabel('Sample Quantiles', S.ax{:});
    set(gca,S.gca{:}); grid on;
    lgd = legend({'Data','Reference'}, 'Location','northeast','FontSize',S.fsLeg); lgd.Box='off';
end

% (d) Bland–Altman
nexttile; hold on;
if isempty(obs_all)
    noData('(d) Bland–Altman', S);
else
    m=(obs_all(:)+pred_all(:))/2; d=(obs_all(:)-pred_all(:));
    v = isfinite(m) & isfinite(d);
    m = m(v); d = d(v);
    safeScatter(m, d, S.ms*0.9, S.c(6,:), 0.6);

    md=mean(d,'omitnan'); sd=std(d,1,'omitnan'); loa=1.96*sd;
    yline(md,'-','Color',S.c(2,:),'LineWidth',S.lw);
    yline(md - loa,'--','Color',S.gray2,'LineWidth',S.lwThin);
    yline(md + loa,'--','Color',S.gray2,'LineWidth',S.lwThin);

    title('(d) Bland–Altman', S.ttl{:});
    xlabel('Mean of Methods', S.ax{:}); ylabel('Difference', S.ax{:});
    set(gca,S.gca{:}); grid on;
    lgd = legend({'Data','Mean','±1.96 SD'}, 'Location','northeast','FontSize',S.fsLeg); lgd.Box='off';
end

exportIEEE(fig1, fullfile(outDir,'figure1_validation'));

%% ================= Figure 2: Mechanism (1×2) =================
fig2 = newFig('double', S);
tiledlayout(fig2,1,2,'TileSpacing','compact','Padding','compact');

% (a) 3D efficiency surface
nexttile; hold on;
Srf = getOrEmpty(T,'Fig2_1_3D_Surface');
if isempty(Srf)
    noData('(a) 3D Surface', S);
else
    [spdCol, prsCol, effCol] = guessSPE(Srf);
    x=Srf.(spdCol); y=Srf.(prsCol); z=Srf.(effCol);
    [Xg,Yg,Zg] = gridTri(x,y,z,160);
    if isempty(Xg)
        noData('(a) 3D Surface', S);
    else
        surf(Xg,Yg,Zg,'EdgeColor','none'); 
        colormap(gca,S.cmap); material dull; shading interp;
        light('Position',[1 0 1],'Style','infinite'); 
        light('Position',[-1 0 1],'Style','infinite');
        contour3(Xg,Yg,Zg,12,'k-','LineWidth',0.6);
        safeScatter3(x,y,z,8,[0 0 0]+0.25,0.25);

        % ridge along x
        try
            xs = linspace(min(Xg(:)),max(Xg(:)),60);
            ys = zeros(size(xs)); zs = zeros(size(xs));
            for ii=1:numel(xs)
                [~,j] = min(abs(Xg(1,:)-xs(ii)));
                [v, irow] = max(Zg(:,j));
                ys(ii) = Yg(irow,j); zs(ii) = v;
            end
            plot3(xs, ys, zs, '-', 'Color', S.c(2,:), 'LineWidth', S.lw+0.3);
        end

        title('(a) 3D Efficiency Surface', S.ttl{:});
        xlabel('Speed (rpm)', S.ax{:}); ylabel('Pressure (kPa)', S.ax{:}); zlabel('Efficiency (%)', S.ax{:});
        set(gca,S.gca3d{:}); view(45,28); box on;
        cb=colorbar; cb.Label.String='Efficiency (%)'; cb.Label.FontSize=S.fsLab; cb.Label.FontName=S.font;
    end
end

% (b) Contour with gradients
nexttile; hold on;
if isempty(Srf)
    noData('(b) Contour', S);
else
    [spdCol, prsCol, effCol] = guessSPE(Srf);
    [Xg,Yg,Zg] = gridTri(Srf.(spdCol), Srf.(prsCol), Srf.(effCol), 200);
    if isempty(Xg)
        noData('(b) Contour', S);
    else
        contourf(Xg,Yg,Zg,18,'LineColor',[0.85 0.85 0.85]);
        colormap(gca,S.cmap);
        [GX,GY] = gradient(Zg, mean(diff(Xg(1,:))), mean(diff(Yg(:,1))));
        step = max(1,round(size(Zg,1)/22));
        quiver(Xg(1:step:end,1:step:end), Yg(1:step:end,1:step:end), ...
               GX(1:step:end,1:step:end), GY(1:step:end,1:step:end), ...
               1.0,'Color',[0 0 0]+0.25,'LineWidth',0.7);

        % ridge projection
        try
            xs = linspace(min(Xg(:)),max(Xg(:)),60);
            ys = zeros(size(xs));
            for ii=1:numel(xs)
                [~,j] = min(abs(Xg(1,:)-xs(ii)));
                [~, irow] = max(Zg(:,j));
                ys(ii) = Yg(irow,j);
            end
            plot(xs, ys, '-', 'Color', S.c(2,:), 'LineWidth', S.lw+0.3);
        end

        title('(b) Contour & Gradient', S.ttl{:});
        xlabel('Speed (rpm)', S.ax{:}); ylabel('Pressure (kPa)', S.ax{:});
        set(gca,S.gca{:});
        cb=colorbar; cb.Label.String='Efficiency (%)'; cb.Label.FontSize=S.fsLab; cb.Label.FontName=S.font;
        lgd = legend({'','Gradient','Ridge'}, 'Location','northeast','FontSize',S.fsLeg); lgd.Box='off';
    end
end

exportIEEE(fig2, fullfile(outDir,'figure2_mechanism'));

%% ================= Figure 3: Dynamics (1×2) =================
fig3 = newFig('double', S);
tiledlayout(fig3,1,2,'TileSpacing','compact','Padding','compact');

% (a) Efficiency time series
nexttile; hold on;
TS = getOrEmpty(T,'Fig5_1_Efficiency_TimeSeries');
if isempty(TS)
    noData('(a) Efficiency over Time', S);
else
    [tcol,ycol] = guessTimeSeries(TS);
    [tt,yy] = sortByTime(TS.(tcol), TS.(ycol));
    h1 = plot(tt,yy,'-','LineWidth',S.lw*0.8,'Color',[S.c(1,:) 0.6]);

    k=max(5, floor(numel(yy)/50)); 
    mu = movmean(yy,k,'omitnan'); 
    sd = movstd(yy,k,'omitnan');
    h2 = fill([tt; flipud(tt)], [mu-2*sd; flipud(mu+2*sd)], S.c(6,:), 'FaceAlpha',0.15, 'EdgeColor','none');
    h3 = plot(tt,mu,'-','LineWidth',S.lw,'Color',S.c(6,:));

    % optional change points
    try
        cp = ischange(yy,'mean','MaxNumChanges',3,'Statistic','mean'); 
        h4 = scatter(tt(cp), yy(cp), 40, S.c(2,:), 'filled');
        handles = [h1 h2 h3 h4]; labs = {'Raw','95% band','Smoothed','Changes'};
    catch
        handles = [h1 h2 h3]; labs = {'Raw','95% band','Smoothed'};
    end

    title('(a) Efficiency over Time', S.ttl{:});
    xlabel('Time Index', S.ax{:}); ylabel('Efficiency (%)', S.ax{:});
    grid on; set(gca,S.gca{:});
    lgd = legend(handles, labs, 'Location','northeast','FontSize',S.fsLeg); lgd.Box='off';
end

% (b) Grouped power time series (columns: time + groups)
nexttile; hold on;
PG = getOrEmpty(T,'Fig5_2_Power_TimeSeries_Grouped');
if isempty(PG)
    noData('(b) Power by Group', S);
else
    if width(PG) >= 2
        t = PG{:,1};
        labels = compose('Config %d', 1:max(1,width(PG)-1));
        L = []; H = {};
        for i = 2:width(PG)
            v = PG{:,i};
            val = isfinite(v) & isfinite(t);
            v = v(val); tt = t(val);
            % gentle cleaning
            if numel(v) > 6
                q1 = quantile(v,0.25); q3 = quantile(v,0.75); iq = q3-q1;
                v(v<q1-3*iq | v>q3+3*iq) = NaN;
            end
            vSm = v;
            if numel(vSm) > 10
                vSm = movmean(vSm, max(5, floor(numel(vSm)/20)), 'omitnan');
            end
            ci = mod(i-2, size(S.c,1))+1;
            plot(tt, v,   '-', 'LineWidth', S.lw*0.4, 'Color', [S.c(ci,:) 0.25]);
            h = plot(tt, vSm, '-', 'LineWidth', S.lw*1.1, 'Color',  S.c(ci,:));
            L = [L, h]; H{end+1} = labels{i-1}; %#ok<AGROW>
        end
        title('(b) Power by Group', S.ttl{:});
        xlabel('Time Index', S.ax{:}); ylabel('Power (kW)', S.ax{:});
        grid on; set(gca,S.gca{:});
        if ~isempty(L)
            lgd = legend(L, H, 'Location','northeast','FontSize',S.fsLeg); lgd.Box='off';
        end
    else
        v = PG{:,1};
        plot(1:numel(v), v, '-', 'LineWidth', S.lw, 'Color', S.c(1,:));
        title('(b) Power by Group', S.ttl{:});
        xlabel('Time Index', S.ax{:}); ylabel('Power (kW)', S.ax{:});
        grid on; set(gca,S.gca{:});
    end
end

exportIEEE(fig3, fullfile(outDir,'figure3_dynamics'));

fprintf('\n✓ Exported figures to: %s\n', outDir);
fprintf('  - PDF, SVG, PNG (600 dpi), EPS\n');
end

%% ================= Style =================
function S = styleIEEE()
S.font   = 'Times New Roman';
S.fsTick = 8;
S.fsLab  = 9;
S.fsTit  = 10;
S.fsLeg  = 9;

S.ttl = {'FontName',S.font,'FontSize',S.fsTit,'FontWeight','bold'};
S.ax  = {'FontName',S.font,'FontSize',S.fsLab};

S.lw = 1.4;
S.lwThin = 1.0;
S.ms = 20;

S.c = [
    0.00,0.45,0.70;  % Blue
    0.90,0.37,0.00;  % Orange
    0.00,0.62,0.45;  % Teal
    0.80,0.47,0.65;  % Purple
    0.95,0.90,0.25;  % Yellow
    0.35,0.70,0.90;  % Light blue
    0.90,0.60,0.00;  % Dark orange
    0.00,0.00,0.00   % Black
];

S.gray2 = [0.55 0.55 0.55];
S.gray3 = [0.35 0.35 0.35];

S.cmap = parula(256);
S.gca  = {'LineWidth',1.0,'Box','off','TickDir','out','Layer','top', ...
          'FontName',S.font,'FontSize',S.fsTick};
S.gca3d = {'LineWidth',1.0,'Box','on','TickDir','out','Layer','top', ...
           'FontName',S.font,'FontSize',S.fsTick};
end

function applyRootStyle(S)
try
  set(groot,'defaultAxesToolbarVisible','off');
  set(groot,'defaultAxesFontName',S.font);
  set(groot,'defaultTextFontName',S.font);
  set(groot,'defaultLegendBox','off');
  set(groot,'defaultLegendFontName',S.font);
catch, end
end

function f = newFig(kind,S)
if strcmpi(kind,'single')
    sz = [3.5 2.625]; % inches
else
    sz = [7.16 4.0];
end
f = figure('Color','w','Units','inches','Position',[1 1 sz], ...
    'PaperUnits','inches','PaperPosition',[0 0 sz],'PaperSize',sz);
end

function exportIEEE(fig, base)
exportgraphics(fig, base+".pdf", 'ContentType','vector','BackgroundColor','white');
exportgraphics(fig, base+".svg", 'ContentType','vector','BackgroundColor','white');
exportgraphics(fig, base+".png", 'Resolution',600,'BackgroundColor','white');
exportgraphics(fig, base+".eps", 'ContentType','vector','BackgroundColor','white');
end

%% ================= CSV / Data utils =================
function var = makeVar(fname)
var = matlab.lang.makeValidName(erase(fname,{'.csv','-'}));
end

function Tbl = readCsvSmart(p)
Tbl = [];
if ~exist(p,'file'), return; end
delims = {',',';','\t'};
for d=1:numel(delims)
  try
    opts = detectImportOptions(p,'Delimiter',delims{d},'NumHeaderLines',0);
    opts.VariableNamingRule = 'preserve';
    opts = setvaropts(opts, opts.VariableNames, 'TreatAsMissing',{'NA','NaN','',' '});
    Tbl = readtable(p, opts);
    if ~isempty(Tbl), break; end
  catch, end
end

% convert stringy numeric columns to double when safe
if ~isempty(Tbl)
  vn = Tbl.Properties.VariableNames;
  for k=1:numel(vn)
    col = Tbl.(vn{k});
    if iscellstr(col) || isstring(col)
      num = str2double(string(col));
      if sum(isfinite(num)) >= max(3, round(0.5*numel(num)))
        Tbl.(vn{k}) = num;
      end
    end
  end
end
end

function t = getOrEmpty(T, field)
f = makeVar(field);
if isfield(T,f), t=T.(f); else, t=[]; end
end

%% ================= Stats helpers =================
function [obs,pred] = findPredObs(Tb)
obs=[]; pred=[];
if isempty(Tb), return; end
low = lower(string(Tb.Properties.VariableNames));
pi = find(contains(low,["pred","yhat","fitted","estimate"]),1);
if isempty(pi), pi=min(2,width(Tb)); end
oi = find(contains(low,["obs","actual","truth","target","y"]),1);
if isempty(oi), oi=1; end
pred = Tb.(Tb.Properties.VariableNames{pi});
obs  = Tb.(Tb.Properties.VariableNames{oi});
if iscellstr(pred)||isstring(pred), t=str2double(string(pred)); if any(isfinite(t)), pred=t; end, end
if iscellstr(obs) ||isstring(obs),  t=str2double(string(obs));  if any(isfinite(t)),  obs=t;  end, end
m = isfinite(obs)&isfinite(pred);
obs=obs(m); pred=pred(m);
end

function [R2,RMSE,MAE,Bias] = metrics(obs,pred)
valid = isfinite(obs) & isfinite(pred);
obs = obs(valid); pred = pred(valid);
C = corrcoef(obs,pred);
r = C(1,2);
R2 = r.^2;
RMSE = sqrt(mean((obs-pred).^2,'omitnan'));
MAE  = mean(abs(obs-pred),'omitnan');
Bias = mean(pred-obs,'omitnan');
end

function [x2,y2] = iqrGate(x,y,k)
if nargin<3, k=3.0; end
X = [x(:),y(:)];
keep = true(size(x));
for j=1:2
  q1 = quantile(X(:,j),0.25);
  q3 = quantile(X(:,j),0.75);
  iq = q3-q1;
  lo = q1-k*iq; hi = q3+k*iq;
  keep = keep & X(:,j)>=lo & X(:,j)<=hi;
end
x2 = x(keep); y2 = y(keep);
end

function [x2,y2] = madGate(x,y,th)
if nargin<3, th=3.5; end
z1 = 0.6745*(x-median(x,'omitnan'))/mad(x,1);
z2 = 0.6745*(y-median(y,'omitnan'))/mad(y,1);
keep = abs(z1)<=th & abs(z2)<=th & isfinite(z1) & isfinite(z2);
x2 = x(keep); y2 = y(keep);
end

function [xin,yin,inliers] = ransacLine(x,y,tol,maxIter)
if nargin<3, tol=0.02; end
if nargin<4, maxIter=400; end
x = x(:); y = y(:); n = numel(x);
if n<3, xin=x; yin=y; inliers=true(n,1); return; end
best = false(n,1); bestN = 0;
for it=1:maxIter
  idx = randperm(n,2);
  ab = [x(idx) ones(2,1)]\y(idx);
  yhat = ab(1)*x+ab(2);
  res = abs(y-yhat);
  thr = tol*max(range(y),eps);
  in = res<=thr;
  if nnz(in)>bestN, bestN=nnz(in); best=in; end
end
if bestN<max(3,round(0.5*n)), inliers=true(n,1); else, inliers=best; end
xin = x(inliers); yin = y(inliers);
end

%% ================= Surface / mapping =================
function [spd,prs,eff] = guessSPE(S)
low = lower(string(S.Properties.VariableNames));
si = find(contains(low,["speed","rpm","s"]),1); if isempty(si), si=1; end
pi = find(contains(low,["press","pressure","p"]),1); if isempty(pi), pi=2; end
ei = find(contains(low,["eff","eta","y"]),1); if isempty(ei), ei=3; end
spd = S.Properties.VariableNames{si};
prs = S.Properties.VariableNames{pi};
eff = S.Properties.VariableNames{ei};
end

function [Xg,Yg,Zg] = gridTri(x,y,z,n)
if nargin<4, n=160; end
ok = isfinite(x)&isfinite(y)&isfinite(z);
x=x(ok); y=y(ok); z=z(ok);
if isempty(x)
    Xg=[]; Yg=[]; Zg=[]; return;
end
F = scatteredInterpolant(x(:),y(:),z(:),'natural','linear');
xr = linspace(min(x),max(x),n);
yr = linspace(min(y),max(y),n);
[Xg,Yg] = meshgrid(xr,yr);
Zg = F(Xg,Yg);
end

%% ================= Time series =================
function [tcol,ycol] = guessTimeSeries(TS)
low = lower(string(TS.Properties.VariableNames));
ti = find(contains(low,["time","t","timestamp","date","index"]),1); if isempty(ti), ti=1; end
yi = find(~contains(low,["time","t","timestamp","date","group","id","label"]),1);
if isempty(yi), yi=min(2,width(TS)); end
tcol = TS.Properties.VariableNames{ti};
ycol = TS.Properties.VariableNames{yi};
end

function [tt,yy,idx] = sortByTime(t,y)
if iscellstr(t)||isstring(t)
  try, tt=datetime(t); catch, tt=(1:numel(t))'; end
else
  tt=t;
end
[tt,idx] = sort(tt);
yy = y(idx);
end

%% ================= Plot helpers =================
function [qqx,qqy] = qqPoints(x)
x = x(isfinite(x));
x = sort(x);
n = numel(x);
p = ((1:n)'-0.5)/n;
qqx = -sqrt(2)*erfcinv(2*p);
qqy = x;
end

function safeScatter(x,y,ms,color,alpha)
try
    scatter(x,y,ms,color,'filled','MarkerFaceAlpha',alpha,'MarkerEdgeColor','none');
catch
    scatter(x,y,ms,color,'filled','MarkerEdgeColor','none');
end
end

function safeScatter3(x,y,z,ms,color,alpha)
try
    scatter3(x,y,z,ms,color,'filled','MarkerFaceAlpha',alpha,'MarkerEdgeColor','none');
catch
    scatter3(x,y,z,ms,color,'filled','MarkerEdgeColor','none');
end
end

function noData(tit,S)
title(tit, S.ttl{:});
text(0.5,0.5,'No data','HorizontalAlignment','center');
axis([0 1 0 1]); set(gca,'XTick',[],'YTick',[]);
end

function confEllipse(x,y,alpha,color,lw)
if nargin<3, alpha=0.95; end
if nargin<4, color=[0 0 0]; end
if nargin<5, lw=1.0; end
x=x(:); y=y(:);
mu = [mean(x,'omitnan'); mean(y,'omitnan')];
C = cov([x y],'omitrows'); if any(~isfinite(C(:))), return; end
[V,D] = eig(C); if any(~isfinite(D(:))) || any(~isfinite(V(:))), return; end
t = linspace(0,2*pi,200);
% chi-square radius for 2 dof
if alpha == 0.95
    k = sqrt(5.991);
elseif alpha == 0.99
    k = sqrt(9.210);
else
    k = sqrt(-2 * log(1 - alpha));
end
E = (V*sqrt(D))*[cos(t); sin(t)]*k;
plot(mu(1)+E(1,:), mu(2)+E(2,:), '-', 'Color', color, 'LineWidth', lw);
end
