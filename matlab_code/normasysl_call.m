% Define parameters with comments
p =  [0.5;           % kp1
      0.1;           % km1
      0.5;           % k2
      (2.13e11)*(1e-9); % kp3
      1.32e1;        % km3
      (6.00e9)*(1e-0);  % kp4 7 Changed e-2, e-7
      3.60e1;        % km5
      (1.2e11)*(1e-9);  % kp5
      4.00e1;        % km5
      1.01e1;        % kp6
      1.01e1;        % km6
      1.02e1;        % k7
      2.70e1;        % k8
      (3.60e9)*(1e-7);  % kp9 % Changed
      3.00e1;        % km9
      1.00e-1;        % k10 +2
      3.85e-2;       % k14
      1.73e-3;       % k15 -1
      2.00e-1;       % v16 crabps synth
      2.00e-1;       % v17 rar synth
      2.00e-2;       % v18 cyp synth
      2.50e-2;       % v19 aldh synth
      3.85e-2;       % k20 crabps deg.
      1.73e-1;       % k21 rar deg
      8.35e-3;       % k22 cyp deg
      1.00e-2;       % k23 aldh deg
      1.00e-1];      % additional parameter
% 

% Set other parameters
retef = 0.0015; % Be careful with making this too large
wntef = 100;
CCret = 1000;
CCwnt = 1000;
funcpercent = 1;
k8log = false ;
k17log = false;

% Define a range of apc_s values to analyze
  %apc_s_vals = [0.25, 0.5, 0.75, 1.0]; 
    %apc_s_vals = [1.5, 1]; 
%apc_s_vals = [1, 10, 20, 30]; % Adjusted values
  %apc_s_vals = [1, 5, 10, 15]; % Adjusted values 
gamma1 = 1;
apc_s = 1;
k16_dynamic = 1;
v18_factor=1;
%axin_s = 0;

 %BcatTcf function
% function val = bctf(x)
%     TCF0 = 15;
%     K16 = 30;
%     w = 1000;
%     val = TCF0 .* x ./ (K16 + w * x);
% end

function val = bctf(x, Ci)
    TCF0 = 15;  
    w = 1000;
    a= 1/0.133;
    r = 1000;
    % Define constants inside the function
    K16_max = 200; %100 150
    K16_0=100;
    Cs = 0.005;%0.001
    nc = 2; %3, 3.5 4.5
    %K16 = K16_max * (1 + ((w*Ci).^nc) ./ (Cs.^nc + (w*Ci).^nc));  % Now K16 depends on Ci dynamically
    K16 = K16_0 + (K16_max .* ((w*Ci).^nc)) ./ (Cs.^nc + ((w*Ci).^nc));
    val = TCF0 .* x ./ (K16 + w * x);
end

%axin_s = 10;

% [t, Sol, treatvals] = modelsys_2025(p, retef, wntef, CCret, CCwnt, funcpercent, apc_s, k8log, k17log);
% %[t, Sol, treatvals] = modelsys_2025_k16dynamic(p, retef, wntef, CCret, CCwnt, funcpercent, apc_s, k8log, k17log);
% APCmin = min(Sol(:,19));
% p(end+1) = APCmin; % Grabs min for uodated linked code
% % Extract Ba and R from Sol
 %Ba = Sol(:, 20);  
% R = Sol(:, 4);  
% H5 = Sol(:, 22);
% P = Sol(:, 19);
% H13 = Sol(:, 26);
% Ca = Sol(:,  25);
% M = Sol(:,  23);
% C = Sol(:, 9);
 %Nr = Sol(:, 8);
% Mi = Sol(:, 24);
% Ci = Sol(:, 27);
% Bp = Sol(:,17);
 %BcatTcf = bctf(Sol(:, 20));
%BcatTcf = bctf(Sol(:, 20), Sol(:, 27));  % Pass Ci (Sol(:, 27)) into bctf
%     % Append BcatTcf to Sol
%     Sol = [Sol BcatTcf];
% 
% % % Compute BcatTcf from the solution
% %     BcatTcf = bctf(Sol(:, 20)); % Assuming BcatTcf is computed from the 20th column of Sol
% % 
% %     % Append BcatTcf to Sol
% %     Sol = [Sol BcatTcf];
% 
% 
% %parameter for end behavior
% %j = find(t>46000);
% 
% %mask = (t >= 4000) & (t <= 10000);
% mask = (t >=20);
% 
% 
% 
% %% HOXA5, HOXA13, MYC, MIZ1 Plot
% fig_HOX = figure;
% 	%darkGreen=[0, 0.5, 0];  % RGB for Green blue deepRed = [0.6, 0, 0]
%   %darkGreen= [0.7, 0.1, 0.1]; red
% darkGreen = [0.1, 0.1, 0.1];
% % --- HOXA5 ---
% subplot(2,2,1);
% plot(t(mask), H5(mask), 'Color', darkGreen, 'LineWidth', 2.75);
% xlabel('time ($\tau$)', 'Interpreter', 'latex');
% ylabel('(non-dim)', 'Interpreter', 'latex', 'FontSize', 24);
% title('HOXA5', 'FontSize', 24, 'Interpreter', 'latex', 'FontWeight', 'normal');
% set(gca, 'FontSize', 27);
% 
% % --- HOXA13 ---
% subplot(2,2,2);
% plot(t(mask), H13(mask), 'Color', darkGreen, 'LineWidth', 2.75);
% xlabel('time ($\tau$)', 'Interpreter', 'latex');
% ylabel('(non-dim)', 'Interpreter', 'latex', 'FontSize', 24);
% title('HOXA13', 'FontSize', 24, 'Interpreter', 'latex', 'FontWeight', 'normal');
% set(gca, 'FontSize', 27);
% 
% % --- MYC ---
% subplot(2,2,3);
% plot(t(mask), M(mask), 'Color', darkGreen, 'LineWidth', 2.75);
% xlabel('time ($\tau$)', 'Interpreter', 'latex');
% ylabel('(non-dim)', 'Interpreter', 'latex', 'FontSize', 24);
% title('MYC', 'FontSize', 24, 'Interpreter', 'latex', 'FontWeight', 'normal');
% set(gca, 'FontSize', 27);
% 
% % --- MIZ1 ---
% subplot(2,2,4);
% plot(t(mask), Mi(mask), 'Color', darkGreen, 'LineWidth', 2.75);
% xlabel('time ($\tau$)', 'Interpreter', 'latex');
% ylabel('(non-dim)', 'Interpreter', 'latex', 'FontSize', 24);
% title('MIZ1', 'FontSize', 24, 'Interpreter', 'latex', 'FontWeight', 'normal');
% set(gca, 'FontSize', 27);
% 
% % --- Figure size and export ---
% set(fig_HOX, 'Position', [100, 100, 1000, 800]);
% exportgraphics(fig_HOX, 'hox_comp_solution_wntoff.pdf', 'ContentType', 'vector');
% 
% 
% %% Complexes: MYC:MIZ1 and HOXA13/β-catenin
% fig_complexes = figure;
% 
% % --- MYC:MIZ1 Complex ---
% subplot(1,2,1);
% plot(t(mask), Ca(mask), 'Color', darkGreen, 'LineWidth', 2.75);
% xlabel('time ($\tau$)', 'Interpreter', 'latex');
% ylabel('(non-dim)', 'Interpreter', 'latex', 'FontSize', 24);
% title('MYC:MIZ1 Complex', 'FontSize', 24, 'Interpreter', 'latex', 'FontWeight', 'normal');
% set(gca, 'FontSize', 27);
% 
% % --- HOXA13/β-catenin Complex ---
% subplot(1,2,2);
% plot(t, Ci, 'Color', darkGreen, 'LineWidth', 2.75);
% xlabel('time ($\tau$)', 'Interpreter', 'latex');
% ylabel('(non-dim)', 'Interpreter', 'latex', 'FontSize', 24);
% title('HOXA13:$\beta$-catenin Complex', 'FontSize', 24, 'Interpreter', 'latex', 'FontWeight', 'normal');
% set(gca, 'FontSize', 27);
% 
% % --- Set figure size and export ---
% set(fig_complexes, 'Position', [100, 100, 1000, 400]);
% exportgraphics(fig_complexes, 'myc_miz1_hoxa13_beta_cat_complexes_wntoff.pdf', 'ContentType', 'vector');


% --- Constant K16 model ---
[t1, Sol1, ~] = modelsys_2025(p, retef, wntef, CCret, CCwnt, funcpercent, apc_s, k8log, k17log);
APCmin = min(Sol1(:,19));
p(end+1) = APCmin;

% Extract state variables
H5_const = Sol1(:, 22);
H13_const = Sol1(:, 26);
M_const = Sol1(:, 23);
Mi_const = Sol1(:, 24);
Ca_const = Sol1(:, 25);
Ci_const = Sol1(:, 27);
Ba_const = Sol1(:, 20);
P_const = Sol1(:, 19);
Nr_const = Sol1(:, 8);
%BcatTcf_const = bctf(Sol1(:, 20)); % Assuming BcatTcf is computed from the 20th column of Sol
Bp_const = Sol1(:,17);
V_const = Sol1(:, 14);
Di_const = Sol1(:, 15);
Db_const = Sol1(:, 16);
Da_const = Sol1(:, 18);

% --- Dynamic K16 model ---
[t2, Sol2, ~] = modelsys_2025_k16dynamic(p, retef, wntef, CCret, CCwnt, funcpercent, k8log, k17log, gamma1);
APCmin = min(Sol2(:,19));
p(end+1) = APCmin;

H5_dyn = Sol2(:, 22);
H13_dyn = Sol2(:, 26);
M_dyn = Sol2(:, 23);
Mi_dyn = Sol2(:, 24);
Ca_dyn = Sol2(:, 25);
Ci_dyn = Sol2(:, 27);
Ba_dyn = Sol2(:, 20);  
Nr_dyn = Sol2(:, 8);
P_dyn = Sol2(:, 19);
%BcatTcf_dyn = bctf(Sol2(:, 20)); % Assuming BcatTcf is computed from the 20th column of Sol
BcatTcf_dyn = bctf(Sol2(:, 20), Sol2(:, 27));
Bp_dyn = Sol2(:,17);
V_dyn = Sol2(:, 14);
Di_dyn = Sol2(:, 15);
Db_dyn = Sol2(:, 16);
Da_dyn = Sol2(:, 18);


mask = (t2 >= 0);  % Assuming both time vectors are the same

fig_HOX = figure;
c_black = [0.1, 0.1, 0.1];       % Constant K16
c_red = [0.6, 0, 0];             % Dynamic K16

% === Create Figure ===
fig_hox_comp = figure;

% --- HOXA5 ---
subplot(2,2,1); hold on; box on;
plot(t1(mask), H5_const(mask), 'Color', c_black, 'LineWidth', 2.5);
plot(t1(mask), H5_dyn(mask), 'Color', c_red, 'LineWidth', 2.5);
title('HOXA5', 'Interpreter', 'latex', 'FontSize', 22, 'FontWeight', 'normal');

% --- HOXA13 ---
subplot(2,2,2); hold on; box on;
plot(t1(mask), H13_const(mask), 'Color', c_black, 'LineWidth', 2.5);
plot(t1(mask), H13_dyn(mask), 'Color', c_red, 'LineWidth', 2.5);
title('HOXA13', 'Interpreter', 'latex', 'FontSize', 22, 'FontWeight', 'normal');

% --- MYC ---
subplot(2,2,3); hold on; box on;
plot(t1(mask), M_const(mask), 'Color', c_black, 'LineWidth', 2.5);
plot(t1(mask), M_dyn(mask), 'Color', c_red, 'LineWidth', 2.5);
title('MYC', 'Interpreter', 'latex', 'FontSize', 22, 'FontWeight', 'normal');

% --- MIZ1 ---
subplot(2,2,4); hold on; box on;
plot(t1(mask), Mi_const(mask), 'Color', c_black, 'LineWidth', 2.5);
plot(t1(mask), Mi_dyn(mask), 'Color', c_red, 'LineWidth', 2.5);
title('MIZ1', 'Interpreter', 'latex', 'FontSize', 22, 'FontWeight', 'normal');

% Label formatting for all subplots
for k = 1:4
    subplot(2,2,k);
    xlabel('time ($\tau$)', 'Interpreter', 'latex');
    ylabel('(non-dim)', 'Interpreter', 'latex', 'FontSize', 22);
    set(gca, 'FontSize', 24);
end

% --- Add legend to one subplot ---
subplot(2,2,4);
lgd = legend({'$K_{16}$ constant', '$K_{16}$ dynamic'}, ...
    'Interpreter', 'latex', 'FontSize', 18, 'Location', 'best');
set(lgd, 'Box', 'on', 'LineWidth', 0.01);

% --- General title ---
sgtitle('\textbf{Comparison of constant vs dynamic $K_{16}$ on HOX pathway components}', ...
    'Interpreter', 'latex', 'FontSize', 24);

% --- Save as vector graphic ---
set(fig_hox_comp, 'Position', [100, 100, 1000, 800]);
%exportgraphics(fig_hox_comp, 'k16_comparison_hox_myc_updated.pdf', 'ContentType', 'vector');

% % --- HOXA5 ---
% subplot(2,2,1); hold on;
% plot(t1(mask), H5_const(mask), 'Color', c_black, 'LineWidth', 2.75);
% plot(t1(mask), H5_dyn(mask), 'Color', c_red, 'LineWidth', 2.75);
% title('HOXA5', 'Interpreter', 'latex', 'FontSize', 24, 'FontWeight', 'normal');
% 
% % --- HOXA13 ---
% subplot(2,2,2); hold on;
% plot(t1(mask), H13_const(mask), 'Color', c_black, 'LineWidth', 2.75);
% plot(t1(mask), H13_dyn(mask), 'Color', c_red, 'LineWidth', 2.75);
% title('HOXA13', 'Interpreter', 'latex', 'FontSize', 24, 'FontWeight', 'normal');
% 
% % --- MYC ---
% subplot(2,2,3); hold on;
% plot(t1(mask), M_const(mask), 'Color', c_black, 'LineWidth', 2.75);
% plot(t1(mask), M_dyn(mask), 'Color', c_red, 'LineWidth', 2.75);
% title('MYC', 'Interpreter', 'latex', 'FontSize', 24, 'FontWeight', 'normal');
% 
% % --- MIZ1 ---
% subplot(2,2,4); hold on;
% plot(t1(mask), Mi_const(mask), 'Color', c_black, 'LineWidth', 2.75);
% plot(t1(mask), Mi_dyn(mask), 'Color', c_red, 'LineWidth', 2.75);
% title('MIZ1', 'Interpreter', 'latex', 'FontSize', 24, 'FontWeight', 'normal');
% 
% % Label formatting for all subplots
% for k = 1:4
%     subplot(2,2,k);
%     xlabel('time ($\tau$)', 'Interpreter', 'latex');
%     ylabel('(non-dim)', 'Interpreter', 'latex', 'FontSize', 24);
%     set(gca, 'FontSize', 27);
% end
% 
% % Legend in MIZ1 subplot
% subplot(2,2,4);
% lgd = legend({'$K_{16}$ constant', '$K_{16}$ dynamic'}, 'Interpreter', 'latex', ...
%              'FontSize', 20, 'Location', 'best');
% set(lgd, 'Box', 'on', 'LineWidth', 0.01);
% 
% set(fig_HOX, 'Position', [100, 100, 1000, 800]);
% exportgraphics(fig_HOX, 'k16_comparison_HOXA_MYC_MIZ1_wntoff.pdf', 'ContentType', 'vector');





fig_wnt_comp = figure;
c_black = [0.1, 0.1, 0.1];       % Constant K16
c_red = [0.6, 0, 0];             % Dynamic K16

% Optional: Adjust mask to be valid for both time arrays
mask1 = (t1 >= 0);
mask2 = (t2 >= 0);

% --- APC ---
subplot(2,2,1); hold on;
plot(t1(mask1), P_const(mask1), 'Color', c_black, 'LineWidth', 2.5);
plot(t2(mask2), P_dyn(mask2), 'Color', c_red, 'LineWidth', 2.5);
title('APC', 'Interpreter', 'latex', 'FontSize', 22, 'FontWeight', 'normal');
box on;

% --- $\beta$-catenin ---
subplot(2,2,2); hold on;
plot(t1(mask1), Ba_const(mask1), 'Color', c_black, 'LineWidth', 2.5);
plot(t2(mask2), Ba_dyn(mask2), 'Color', c_red, 'LineWidth', 2.5);
title('$\beta$-catenin', 'Interpreter', 'latex', 'FontSize', 22, 'FontWeight', 'normal');
box on;

% --- $\beta$-catenin:TCF complex ---
subplot(2,2,3); hold on;
plot(t1(mask1), BcatTcf_const(mask1), 'Color', c_black, 'LineWidth', 2.5);
plot(t2(mask2), BcatTcf_dyn(mask2), 'Color', c_red, 'LineWidth', 2.5);
title('$\beta$-cat:TCF', 'Interpreter', 'latex', 'FontSize', 22, 'FontWeight', 'normal');
box on;

% --- Phospho-$\beta$-catenin ---
subplot(2,2,4); hold on;
plot(t1(mask1), Bp_const(mask1), 'Color', c_black, 'LineWidth', 2.5);
plot(t2(mask2), Bp_dyn(mask2), 'Color', c_red, 'LineWidth', 2.5);
title('Phospho-$\beta$-cat', 'Interpreter', 'latex', 'FontSize', 22, 'FontWeight', 'normal');
box on;

% === Common Axis Labels and Font ===
for k = 1:4
    subplot(2,2,k);
    xlabel('time ($\tau$)', 'Interpreter', 'latex', 'FontSize', 22);
    ylabel('(non-dim)', 'Interpreter', 'latex', 'FontSize', 22);
    set(gca, 'FontSize', 22);
end

% === Add legend only in bottom-right ===
subplot(2,2,4);
lgd = legend({'$K_{16}$ constant', '$K_{16}$ dynamic'}, 'Interpreter', 'latex', ...
             'FontSize', 20, 'Location', 'best');
set(lgd, 'Box', 'on', 'LineWidth', 0.01);

% === General Title ===
sgtitle('\textbf{Comparison of constant vs dynamic $K_{16}$ on WNT pathway components}', ...
        'Interpreter', 'latex', 'FontSize', 24);

% === Export as PDF ===
set(fig_wnt_comp, 'Position', [100, 100, 1000, 800]);
%exportgraphics(fig_wnt_comp, 'k16_comparison_wnt_comp_updated.pdf', 'ContentType', 'vector');


% % --- APC ---
% subplot(2,2,1); hold on;
% plot(t1(mask1), P_const(mask1), 'Color', c_black, 'LineWidth', 2.5);
% plot(t2(mask2), P_dyn(mask2), 'Color', c_red, 'LineWidth', 2.5);
% title('APC', 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'normal');
% 
% % --- $\beta$-catenin ---
% subplot(2,2,2); hold on;
% plot(t1(mask1), Ba_const(mask1), 'Color', c_black, 'LineWidth', 2.5);
% plot(t2(mask2), Ba_dyn(mask2), 'Color', c_red, 'LineWidth', 2.5);
% title('$\beta$-catenin', 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'normal');
% 
% % --- $\beta$-catenin:TCF complex ---
% subplot(2,2,3); hold on;
% plot(t1(mask1), BcatTcf_const(mask1), 'Color', c_black, 'LineWidth', 2.5);
% plot(t2(mask2), BcatTcf_dyn(mask2), 'Color', c_red, 'LineWidth', 2.5);
% title('$\beta$-cat:TCF', 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'normal');
% 
% % --- Phospho-$\beta$-catenin ---
% subplot(2,2,4); hold on;
% plot(t1(mask1), Bp_const(mask1), 'Color', c_black, 'LineWidth', 2.5);
% plot(t2(mask2), Bp_dyn(mask2), 'Color', c_red, 'LineWidth', 2.5);
% title('Phospho-$\beta$-cat', 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'normal');
% 
% % Label formatting for all subplots
% for k = 1:4
%     subplot(2,2,k);
%     xlabel('time ($\tau$)', 'Interpreter', 'latex');
%     ylabel('(non-dim)', 'Interpreter', 'latex', 'FontSize', 22);
%     set(gca, 'FontSize', 24);
% end

% Legend
% subplot(2,2,4);
% lgd = legend({'$K_{16}$ constant', '$K_{16}$ dynamic'}, 'Interpreter', 'latex', ...
%              'FontSize', 18, 'Location', 'best');
% set(lgd, 'Box', 'on', 'LineWidth', 0.01);
% 
% % Fix figure handle and save
% set(fig_wnt_comp, 'Position', [100, 100, 1000, 800]);
% exportgraphics(fig_wnt_comp, 'k16_comparison_wnt_comp.pdf', 'ContentType', 'vector');



%plot(t1(mask), Nr_const(mask), 'LineWidth', 2.75);

% --- WNT Core Components with Constant Binding Rate ---
  % Truncate early transient values

% fig_WNT = figure;
% c_black = [0.1, 0.1, 0.1];  % Nice black for constant-case plots

% % --- APC (P_const) ---
% subplot(2,2,1); hold on;
% plot(t1(mask), P_const(mask), 'Color', c_black, 'LineWidth', 2.5);
% title('APC', 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'normal');
% 
% % --- Beta-catenin (Ba_const) ---
% subplot(2,2,2); hold on;
% plot(t1(mask), Ba_const(mask), 'Color', c_black, 'LineWidth', 2.5);
% title('$\beta$-catenin', 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'normal');
% 
% % --- Beta-catenin:TCF complex (BcatcF_const) ---
% subplot(2,2,3); hold on;
% plot(t1(mask), BcatTcf_const(mask), 'Color', c_black, 'LineWidth', 2.5);
% title('$\beta$-cat:TCF', 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'normal');
% 
% % --- Phospho Beta-catenin (Bp_const) ---
% subplot(2,2,4); hold on;
% plot(t1(mask), Bp_const(mask), 'Color', c_black, 'LineWidth', 2.5);
% title('Phospho-$\beta$-cat', 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'normal');
% 
% % Axis and font formatting
% for k = 1:4
%     subplot(2,2,k);
%     xlabel('time ($\tau$)', 'Interpreter', 'latex');
%     ylabel('(non-dim)', 'Interpreter', 'latex', 'FontSize', 22);
%     set(gca, 'FontSize', 25);
%     box on;
% end
% 
% set(fig_WNT, 'Position', [100, 100, 1000, 800]);
% exportgraphics(fig_WNT, 'wnt_comp_const.pdf', 'ContentType', 'vector');

% fig_destr = figure;
% % --- APC (P_const) ---
% subplot(2,2,1); hold on;
% plot(t1(mask), V_const(mask), 'Color', c_black, 'LineWidth', 2.5);
% title('Destruction Complex', 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'normal');
% 
% % --- Beta-catenin (Ba_const) ---
% subplot(2,2,2); hold on;
% plot(t1(mask), Db_const(mask), 'Color', c_black, 'LineWidth', 2.5);
% title('Dishevelled', 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'normal');
% 
% % --- Beta-catenin:TCF complex (BcatcF_const) ---
% subplot(2,2,3); hold on;
% plot(t1(mask), Di_const(mask), 'Color', c_black, 'LineWidth', 2.5);
% title('Inactive Destruction Complex ', 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'normal');
% 
% % --- Phospho Beta-catenin (Bp_const) ---
% subplot(2,2,4); hold on;
% plot(t1(mask), Da_const(mask), 'Color', c_black, 'LineWidth', 2.5);
% title('Active Destruction Complex', 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'normal');
% 
% % Axis and font formatting
% for k = 1:4
%     subplot(2,2,k);
%     xlabel('time ($\tau$)', 'Interpreter', 'latex');
%     ylabel('(non-dim)', 'Interpreter', 'latex', 'FontSize', 22);
%     set(gca, 'FontSize', 24);
%     box on;
% end
% 
% set(fig_destr, 'Position', [100, 100, 1000, 800]);
% exportgraphics(fig_destr, 'destruction_comp_const.pdf', 'ContentType', 'vector');
% 

%fig_WNT_dyn = figure;
c_black = [0.1, 0.1, 0.1];  % Nice black for constant-case plots

% % --- APC (P_const) ---
% subplot(2,2,1); hold on;
% plot(t2(mask), P_dyn(mask), 'Color', c_black, 'LineWidth', 2.5);
% title('APC', 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'normal');
% 
% % --- Beta-catenin (Ba_const) ---
% subplot(2,2,2); hold on;
% plot(t2(mask), Ba_dyn(mask), 'Color', c_black, 'LineWidth', 2.5);
% title('$\beta$-catenin', 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'normal');
% 
% % --- Beta-catenin:TCF complex (BcatcF_const) ---
% subplot(2,2,3); hold on;
% plot(t2(mask), BcatTcf_dyn(mask), 'Color', c_black, 'LineWidth', 2.5);
% title('$\beta$-cat:TCF', 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'normal');
% 
% % --- Phospho Beta-catenin (Bp_const) ---
% subplot(2,2,4); hold on;
% plot(t2(mask), Bp_dyn(mask), 'Color', c_black, 'LineWidth', 2.5);
% title('Phospho-$\beta$-cat', 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'normal');
% 
% % Axis and font formatting
% for k = 1:4
%     subplot(2,2,k);
%     xlabel('time ($\tau$)', 'Interpreter', 'latex');
%     ylabel('(non-dim)', 'Interpreter', 'latex', 'FontSize', 22);
%     set(gca, 'FontSize', 25);
%     box on;
% end
% 
% set(fig_WNT_dyn, 'Position', [100, 100, 1000, 800]);
% exportgraphics(fig_WNT_dyn, 'wnt_comp_dyn.pdf', 'ContentType', 'vector');
%%
% % === Create Figure ===
% fig_WNT_const = figure;
% c_black = [0.1, 0.1, 0.1];
% 
% % --- APC ---
% subplot(2,2,1); hold on; box on;
% plot(t1(mask), P_const(mask), 'Color', c_black, 'LineWidth', 2.5);
% title('APC', 'Interpreter', 'latex', 'FontSize', 22, 'FontWeight', 'normal');
% 
% % --- Beta-catenin ---
% subplot(2,2,2); hold on; box on;
% plot(t1(mask), Ba_const(mask), 'Color', c_black, 'LineWidth', 2.5);
% title('$\beta$-catenin', 'Interpreter', 'latex', 'FontSize', 22, 'FontWeight', 'normal');
% 
% % --- Beta-catenin:TCF complex ---
% subplot(2,2,3); hold on; box on;
% plot(t1(mask), BcatTcf_const(mask), 'Color', c_black, 'LineWidth', 2.5);
% title('$\beta$-cat:TCF', 'Interpreter', 'latex', 'FontSize', 22, 'FontWeight', 'normal');
% 
% % --- Phospho-Beta-catenin ---
% subplot(2,2,4); hold on; box on;
% plot(t1(mask), Bp_const(mask), 'Color', c_black, 'LineWidth', 2.5);
% title('Phospho-$\beta$-cat', 'Interpreter', 'latex', 'FontSize', 22, 'FontWeight', 'normal');
% 
% % Axis formatting for all subplots
% for k = 1:4
%     subplot(2,2,k);
%     xlabel('time ($\tau$)', 'Interpreter', 'latex');
%     ylabel('(non-dim)', 'Interpreter', 'latex', 'FontSize', 20);
%     set(gca, 'FontSize', 24);
% end
% 
% % --- General title ---
% sgtitle('\textbf{Dynamics of WNT Pathway Components with Constant $K_{16}$}', ...
%     'Interpreter', 'latex', 'FontSize', 24);
% 
% % --- Save figure ---
% set(fig_WNT_const, 'Position', [100, 100, 1000, 800]);
% exportgraphics(fig_WNT_const, 'wnt_comp_const_updated_wnton.pdf', 'ContentType', 'vector');

%% === Create Figure ===
fig_WNT_const1 = figure;
c_black = [0.1, 0.1, 0.1];
shade_color = [0.96, 0.96, 0.96];

shade_start = 3000;
shade_end = 25000;

% --- APC ($P$) ---
subplot(2,2,1); hold on; box on;
ydata = P_dyn(mask);
plot(t1(mask), ydata, 'Color', c_black, 'LineWidth', 2.5);
yl = ylim;
fill([shade_start shade_end shade_end shade_start], ...
     [yl(1) yl(1) yl(2) yl(2)], shade_color, 'EdgeColor', 'none');
uistack(findobj(gca,'Type','patch'),'bottom');
title('APC ($P$)', 'Interpreter', 'latex', 'FontSize', 20, 'FontWeight', 'normal');

% --- B_a ---
subplot(2,2,2); hold on; box on;
ydata = Ba_dyn(mask);
plot(t1(mask), ydata, 'Color', c_black, 'LineWidth', 2.5);
yl = ylim;
fill([shade_start shade_end shade_end shade_start], ...
     [yl(1) yl(1) yl(2) yl(2)], shade_color, 'EdgeColor', 'none');
uistack(findobj(gca,'Type','patch'),'bottom');
title('$\beta$-catenin ($B_a$)', 'Interpreter', 'latex', 'FontSize', 20, 'FontWeight', 'normal');

% --- B_t ($\beta$-cat:TCF) ---
subplot(2,2,3); hold on; box on;
ydata = BcatTcf_dyn(mask);
plot(t1(mask), ydata, 'Color', c_black, 'LineWidth', 2.5);
yl = ylim;
fill([shade_start shade_end shade_end shade_start], ...
     [yl(1) yl(1) yl(2) yl(2)], shade_color, 'EdgeColor', 'none');
uistack(findobj(gca,'Type','patch'),'bottom');
title('$\beta$-cat:TCF ($B_t$)', 'Interpreter', 'latex', 'FontSize', 20, 'FontWeight', 'normal');

% --- B_p (Phospho-Beta-catenin) ---
subplot(2,2,4); hold on; box on;
ydata = Bp_dyn(mask);
plot(t1(mask), ydata, 'Color', c_black, 'LineWidth', 2.5);
yl = ylim;
fill([shade_start shade_end shade_end shade_start], ...
     [yl(1) yl(1) yl(2) yl(2)], shade_color, 'EdgeColor', 'none');
uistack(findobj(gca,'Type','patch'),'bottom');
title('Phospho-$\beta$-cat ($B_p$)', 'Interpreter', 'latex', 'FontSize', 20, 'FontWeight', 'normal');

% Axes formatting
for k = 1:4
    subplot(2,2,k);
    xlabel('time ($\tau$)', 'Interpreter', 'latex');
    ylabel('(non-dim)', 'Interpreter', 'latex', 'FontSize', 20);
    set(gca, 'FontSize', 18.5);
end

% --- General title ---
sgtitle('\textbf{Dynamics of WNT Pathway Components under Transient WNT Input}', ...
    'Interpreter', 'latex', 'FontSize', 24);

% --- Save figure ---
set(fig_WNT_const1, 'Position', [100, 100, 1000, 800]);
%exportgraphics(fig_WNT_const1, 'wnt_comp_dyn_shade.pdf', 'ContentType', 'vector');

%% Constant K16 hox pathway solutions:
% === Create Figure ===
fig_components_const = figure;

% --- HOXA5 ---
subplot(2,3,1); hold on; box on;
plot(t1(mask), H5_const(mask), 'Color', c_black, 'LineWidth', 2.5);
title('HOXA5 ($H_a$) ', 'Interpreter', 'latex', 'FontSize', 20, 'FontWeight', 'normal');

% --- HOXA13 ---
subplot(2,3,2); hold on; box on;
plot(t1(mask), H13_const(mask), 'Color', c_black, 'LineWidth', 2.5);
title('HOXA13 ($H_i$)', 'Interpreter', 'latex', 'FontSize', 20, 'FontWeight', 'normal');

% --- MYC ---
subplot(2,3,3); hold on; box on;
plot(t1(mask), M_const(mask), 'Color', c_black, 'LineWidth', 2.5);
title('MYC ($M$)', 'Interpreter', 'latex', 'FontSize', 20, 'FontWeight', 'normal');

% --- MIZ1 ---
subplot(2,3,4); hold on; box on;
plot(t1(mask), Mi_const(mask), 'Color', c_black, 'LineWidth', 2.5);
title('MIZ1 ($M_i$)', 'Interpreter', 'latex', 'FontSize', 20, 'FontWeight', 'normal');

% --- MYC:MIZ1 ---
subplot(2,3,5); hold on; box on;
plot(t1(mask), Ca_const(mask), 'Color', c_black, 'LineWidth', 2.5);
title('MYC:MIZ1 ($C_a$)', 'Interpreter', 'latex', 'FontSize', 20, 'FontWeight', 'normal');

% --- HOXA13:$\beta$-cat ---
subplot(2,3,6); hold on; box on;
plot(t1(mask), Ci_const(mask), 'Color', c_black, 'LineWidth', 2.5);
title('HOXA13:$\beta$-cat ($C_i$)', 'Interpreter', 'latex', 'FontSize', 20, 'FontWeight', 'normal');

% Axis formatting
for k = 1:6
    subplot(2,3,k);
    xlabel('time ($\tau$)', 'Interpreter', 'latex');
    ylabel('(non-dim)', 'Interpreter', 'latex', 'FontSize', 20);
    set(gca, 'FontSize', 18);
end

% --- General title ---
sgtitle('\textbf{HOX Components Solution under Constant $K_{16}$ with Permanent WNT}', ...
    'Interpreter', 'latex', 'FontSize', 24);

% --- Save figure ---
set(fig_components_const, 'Position', [100, 100, 1200, 800]);
%exportgraphics(fig_components_const, 'components_const_K16_wnton11.pdf', 'ContentType', 'vector');




%% Constant K16 hox pathway solutions:
% === Create Figure ===
fig_components_const = figure;

shade_color = [0.95, 0.95, 0.95];  % Light gray
shade_start = 3000;
shade_end = 25000;

% --- HOXA5 ---
subplot(2,3,1); hold on; box on;
plot(t1(mask), H5_const(mask), 'Color', c_black, 'LineWidth', 2.5);
yl = ylim;
fill([shade_start shade_end shade_end shade_start], ...
     [yl(1) yl(1) yl(2) yl(2)], shade_color, 'EdgeColor', 'none');
uistack(findobj(gca,'Type','patch'),'bottom');
title('HOXA5 ($H_a$)', 'Interpreter', 'latex', 'FontSize', 12, 'FontWeight', 'normal');

% --- HOXA13 ---
subplot(2,3,2); hold on; box on;
plot(t1(mask), H13_const(mask), 'Color', c_black, 'LineWidth', 2.5);
yl = ylim;
fill([shade_start shade_end shade_end shade_start], ...
     [yl(1) yl(1) yl(2) yl(2)], shade_color, 'EdgeColor', 'none');
uistack(findobj(gca,'Type','patch'),'bottom');
title('HOXA13 ($H_i$)', 'Interpreter', 'latex', 'FontSize', 15, 'FontWeight', 'normal');

% --- MYC ---
subplot(2,3,3); hold on; box on;
plot(t1(mask), M_const(mask), 'Color', c_black, 'LineWidth', 2.5);
yl = ylim;
fill([shade_start shade_end shade_end shade_start], ...
     [yl(1) yl(1) yl(2) yl(2)], shade_color, 'EdgeColor', 'none');
uistack(findobj(gca,'Type','patch'),'bottom');
title('MYC ($M$)', 'Interpreter', 'latex', 'FontSize', 15, 'FontWeight', 'normal');

% --- MIZ1 ---
subplot(2,3,4); hold on; box on;
plot(t1(mask), Mi_const(mask), 'Color', c_black, 'LineWidth', 2.5);
yl = ylim;
fill([shade_start shade_end shade_end shade_start], ...
     [yl(1) yl(1) yl(2) yl(2)], shade_color, 'EdgeColor', 'none');
uistack(findobj(gca,'Type','patch'),'bottom');
title('MIZ1 ($M_i$)', 'Interpreter', 'latex', 'FontSize', 15, 'FontWeight', 'normal');

% --- MYC:MIZ1 ---
subplot(2,3,5); hold on; box on;
plot(t1(mask), Ca_const(mask), 'Color', c_black, 'LineWidth', 2.5);
yl = ylim;
fill([shade_start shade_end shade_end shade_start], ...
     [yl(1) yl(1) yl(2) yl(2)], shade_color, 'EdgeColor', 'none');
uistack(findobj(gca,'Type','patch'),'bottom');
title('MYC:MIZ1 ($C_a$)', 'Interpreter', 'latex', 'FontSize', 15, 'FontWeight', 'normal');

% --- HOXA13:$\beta$-cat ---
subplot(2,3,6); hold on; box on;
plot(t1(mask), Ci_const(mask), 'Color', c_black, 'LineWidth', 2.5);
yl = ylim;
fill([shade_start shade_end shade_end shade_start], ...
     [yl(1) yl(1) yl(2) yl(2)], shade_color, 'EdgeColor', 'none');
uistack(findobj(gca,'Type','patch'),'bottom');
title('HOXA13:$\beta$-cat ($C_i$)', 'Interpreter', 'latex', 'FontSize', 15, 'FontWeight', 'normal');

% Axis formatting
for k = 1:6
    subplot(2,3,k);
    xlabel('time ($\tau$)', 'Interpreter', 'latex');
    ylabel('(non-dim)', 'Interpreter', 'latex', 'FontSize', 20);
    set(gca, 'FontSize', 18.5);
end

% --- General title ---
sgtitle('\textbf{Hox Components Solution under Constant $K_{16}$}', ...
    'Interpreter', 'latex', 'FontSize', 24);

% --- Save figure ---
set(fig_components_const, 'Position', [100, 100, 1200, 800]);
%exportgraphics(fig_components_const, 'components_const_K16_wnton1.pdf', 'ContentType', 'vector');


%% Dynamic K16 hox components solution:
% === Create Figure ===
fig_components_dyn = figure;

% --- HOXA5 ---
subplot(2,3,1); hold on; box on;
plot(t1(mask), H5_dyn(mask), 'Color', c_red, 'LineWidth', 2.5);
title('HOXA5 ($H_a$)', 'Interpreter', 'latex', 'FontSize', 15, 'FontWeight', 'normal');

% --- HOXA13 ---
subplot(2,3,2); hold on; box on;
plot(t1(mask), H13_dyn(mask), 'Color', c_red, 'LineWidth', 2.5);
title('HOXA13 ($H_i$)', 'Interpreter', 'latex', 'FontSize', 15, 'FontWeight', 'normal');

% --- MYC ---
subplot(2,3,3); hold on; box on;
plot(t1(mask), M_dyn(mask), 'Color', c_red, 'LineWidth', 2.5);
title('MYC ($M$)', 'Interpreter', 'latex', 'FontSize', 15, 'FontWeight', 'normal');

% --- MIZ1 ---
subplot(2,3,4); hold on; box on;
plot(t1(mask), Mi_dyn(mask), 'Color', c_red, 'LineWidth', 2.5);
title('MIZ1 ($M_i$)', 'Interpreter', 'latex', 'FontSize', 15, 'FontWeight', 'normal');

% --- MYC:MIZ1 ---
subplot(2,3,5); hold on; box on;
plot(t1(mask), Ca_dyn(mask), 'Color', c_red, 'LineWidth', 2.5);
title('MYC:MIZ1 ($C_a$)', 'Interpreter', 'latex', 'FontSize', 15, 'FontWeight', 'normal');

% --- HOXA13:$\beta$-cat ---
subplot(2,3,6); hold on; box on;
plot(t1(mask), Ci_dyn(mask), 'Color', c_red, 'LineWidth', 2.5);
title('HOXA13:$\beta$-cat ($C_i$)', 'Interpreter', 'latex', 'FontSize', 15, 'FontWeight', 'normal');

% Axis formatting
for k = 1:6
    subplot(2,3,k);
    xlabel('time ($\tau$)', 'Interpreter', 'latex');
    ylabel('(non-dim)', 'Interpreter', 'latex', 'FontSize', 20);
    set(gca, 'FontSize', 18.5);
end

% --- General title ---
sgtitle('\textbf{Dynamic Response of HOX Components to Transient WNT Activation}', ...
    'Interpreter', 'latex', 'FontSize', 24);

% --- Save figure ---
set(fig_components_dyn, 'Position', [100, 100, 1200, 800]);
%exportgraphics(fig_components_dyn, 'components_dyn_K16_wntooon.pdf', 'ContentType', 'vector');

%% Dynamic K16 hox components solution:
% === Create Figure ===
fig_components_dyn = figure;

shade_color = [0.95, 0.95, 0.95];  % Light gray
shade_start = 3000;
shade_end = 25000;

% Data vectors and titles for clean looping
data = {H5_dyn, H13_dyn, M_dyn, Mi_dyn, Ca_dyn, Ci_dyn};
titles = { ...
    'HOXA5 ($H_a$)', ...
    'HOXA13 ($H_i$)', ...
    'MYC ($M$)', ...
    'MIZ1 ($M_i$)', ...
    'MYC:MIZ1 ($C_a$)', ...
    'HOXA13:$\beta$-cat ($C_i$)'};

for k = 1:6
    subplot(2, 3, k); hold on; box on;

    % Extract and plot data
    ydata = data{k}(mask);
    plot(t1(mask), ydata, 'Color', c_red, 'LineWidth', 2.5);

    % Determine data range with buffer
    ymin = min(ydata); ymax = max(ydata); pad = 0.05 * (ymax - ymin);

    % Add shaded region
    h_fill = fill([shade_start shade_end shade_end shade_start], ...
                  [ymin - pad, ymin - pad, ymax + pad, ymax + pad], ...
                  shade_color, 'EdgeColor', 'none');
    uistack(h_fill, 'bottom');

    % Title and formatting
    title(titles{k}, 'Interpreter', 'latex', 'FontSize', 15, 'FontWeight', 'normal');
    xlabel('time ($\tau$)', 'Interpreter', 'latex');
    ylabel('(non-dim)', 'Interpreter', 'latex', 'FontSize', 20);
    set(gca, 'FontSize', 18.5);
end

% --- General title ---
sgtitle('\textbf{Dynamic Response of HOX Components to Transient WNT Activation}', ...
    'Interpreter', 'latex', 'FontSize', 24);

% --- Save figure ---
set(fig_components_dyn, 'Position', [100, 100, 1200, 800]);
%exportgraphics(fig_components_dyn, 'components_dyn_K16_wntooon.pdf', 'ContentType', 'vector');
