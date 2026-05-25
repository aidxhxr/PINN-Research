function [t, Sol, treatvals] = modelsys_2025_k16dynamic(p, retef, wntef, CCret, CCwnt,funcpercent, k8log, k17log, gamma1)
%Description of Inputs
%function [t, Sol] = nd_wnt_retinoid_hox_simp_updated(p, retef, wntef, CCret, CCwnt, funcpercent, apc_s, k8log, k17log, APCmin = min(Sol(:,19)))
%p(end+1) = APCmin;

%p = parameter vector for retinoid pathway
%retef = ret on wnt param, wntef = wnt on ret param
%RSS = RA SS value based on simplified pathway run with parameter set p and 
%link %No longer used!
%CCret = char conc for retinoid, CCwnt = char conc for wnt
%funcpercent = apc functioning, apc_s = multiplier of axin synth
%k8log = if k8 affected is T, k17log = if k17 affected is T

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Time discretization: 
N_t = 5e3;
t_0 = 0; 
t_N = 30000;
time = linspace(t_0,t_N,N_t);

%%%%%%%%%%%%%%%%%%Wnt Parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dimensional Parameters:
DSH0 = 100;
TCF0 = 15;                                                                                                                                                                                                    
APC0 = 100;
GSK0 = 50;



%Use logic input to determine if value of K8 or K17 change
if k8log
    K8 = (1+(1-funcpercent))*120;
else
    K8 = 120;
end

if k17log
    K17 = (1+(1-funcpercent))*1200;
else
    K17 = 1200;
end

K16_max= 200*gamma1; %100 150
K16_0 = 100;%0.1
Cs = 0.005; % 0.005
nc = 2; % 2 3, 3.5 4.5
K7  = 50;
%K16 = 30;
K20 = 1;
K21 = 1;
Km  = 98;

k1  = 0.182; 
k2  = 1.82e-2;
k3  = 5e-2;
k4  = 0.267;
k5  = 0.133;
k6  = 9.09e-2;
k_6 = 0.909; 
k9  = 206;
k10 = 206;
k11 = 0.417;
v12 = 0.423;
k13 = 2.57e-4;
v14 = (8.22e-5);%*(1+0.01*apc_s); %increases synthesis of Axin by given percent
k15 = .33;

% APC Regulating Function Parameters:
Parameters = [212.8453 39.9102 34.1111];
k19 = 1/K17;
v18 = Parameters(1)*k19;
Kt  = Parameters(2);
Kb  = Parameters(3);

eta = 1;
if funcpercent < 1
    v18 = v18*(eta*funcpercent);
end

%% Non-Dimensional Parameters:
w = CCwnt; %conc. scaling for wnt going to be fed in like retinoid conc. scaling.
gsk0 = GSK0/w;
tcf0 = TCF0/w;
dsh0 = DSH0/w;

K7n  = w/K7;
K8n  = w/K8;
%K16n = K16*w;
K17n = w/K17;
%K16p = K16/K17;
K20n = w/K20;
%K21n = K21/K7;
%Kmn  = Km/K17;
Ktn  = (TCF0*w)/Kt;
Kbn  = w/Kb;

k1n  = k1/k5;
k2n  = k2/k5;
k3n  = k3*w/k5;
k4n  = k4/k5;
k6n  = (k6*K21*w^2)/(k5*K7);
k_6n = k_6/k5;
k9n  = (k9*w)/(k5*K8);
k10n = k10/k5;
%v10n = k10*k10n/k11;
k11n = k11/k5;
v12n = v12/(w*k5);
k13n = k13/k5;
v14n = v14/(k5*w);
%v14a = v14/K21/k5;
%v14b = v14/K8/k5;
%v14p = v14/K17/k5;
%k14n = k9*v14b/k5;
k15n = k15*w/k5;
v18n = v18/(w*k5);
k19n = k19/k5;

%%%%%%%%%%%%%%%%%%RAS Parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rkp1 = p(1);
rkm1 = p(2);
rk2 = p(3);
rkp3 = p(4);
rkm3 = p(5);
rkp4 = p(6);
rkm4 = p(7);
rkp5 = p(8);
rkm5 = p(9);
rkp6 = p(10);
rkm6 = p(11);
rk7 = p(12);
rk8 = p(13);
rkp9 = p(14);
rkm9 = p(15);
rk10 = p(16);
rk11 = 0; rk12 = 0; rk13 = 0;
rk14 = p(17);
rk15 = p(18);
rv16 = p(19);
rv17 = p(20);
rv18 = p(21);
rv19 = p(22);
rk20 = p(23);
rk21 = p(24);
rk22 = p(25);
rk23 = p(26);
MCsynth = p(27);

%%%%%%%%%%%%%%%%%%%%%%%%%%% HOX Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters from the HOX model
k31 = 8.25*16.8182e2;% 0.12*1.6182e2 0.18  original = 0.25 2.25 30
%k31 = 0.18*10.8182e2;
%k31 = 0.12;
k32 = 20.85*1.25e1; %0.1    original 0.45*1.25e1 5.85
k33 = 0.30*1.95e2; % 0.1, 0.2
k34 = 0.25*10; % Can be tuned 0.25*10
k36 = 0.13; %0.15
kp37 = 0.45*1.0e-1; %0.45
%kp37 = 0.45*5.0e-2; %0.45
%kp37 = 0.25;
km37 = 0.1*1.65;% 0.1*1.65 .5, 1.65
%km37 = 0.09*1.65;%1.5, 1.65
v31 = 0.15*3.333e1;
%v31 = 0.15;
k38 = 0.083 ;%0.10 0.09
v32 = 0.20*3.333e1;%0.20*3.333e1 2*3.333e1
%v32 = 0.20;
k39 = 0.15; %0.15

%k35 = 0.50; %0.10

%%%%%%%%%%%%


%% HOXA5 deg, and MYC synth parameters
a2 = 7.735; 
%a3 = 1.5;7.5, 19.5;
%a3 = 70.5;
a3 = 50.5; %50.5 45.555
a4 = 1.5;
Kc = 1.0;
a1 = 55.5;
Kd = 0.01; %20.5 0.005
k33_max = 1.0; % 10 100 200 500
n = 1; % 2 original 1
km = 0.25; %0.5 0.25

%%%%%%%%%%%%%%%%%%%%%%%
%New constants
k = 1500;
a5 = 1;
a6 = 1;
n1 = 1; %3
n2 = 1; %4
n3 = 1;
knn1 = 200.5;
knn2 = 100.5;

phi = 0.00139;


k33 = @(Nr) (k33_max *(w*Nr).^n)/(Kd^n + (w*Nr).^n); % new synth of HOXA13
% Nondimensional parameters of HOX model scaled by Bcat
k31n = k31 / (w*k5);
k32n = k32 / k5;
k33n = @(Nr) k33(Nr) / k5;
%k33n =  k33/ k5;
k36n = k36 / k5;
kp37n = kp37 * w / k5;
km37n = km37 / k5;
v31n =  v31 / (w*k5);
k38n = k38 / k5;
v32n = v32 / (w*k5) ;
k39n = k39 / k5;
%k35n = k35/k5;
knn = k/(w*k5);


% Linked parameters
kp40 = 2.5 * 1e-6;%6
%kp40 = 3.6 * 1e-5;
km40 = 1.0 * 1e-3; %3 H13/beta blows up from 10^-5 and larger frequencies for 10^-1
%km40 = 1.0 * 1e-6;
k41 = 0.1475*(1e-4); % 0.15, 0.08 % Controls the oscillation and beta-catenin conc
%k41 = 0.10*(1e-4); % w = 500

% Added non-dim rate constant (linked parameters)
kp40n = (kp40 * w) / k5;
%kp40n = kp40 / k5;
km40n = km40 / k5; % this is important to H13/beta complex
k41n = (k41 * w) / k5 ;
%k41n = k41 / k5 ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%For NonDim purposes of Both pathways
r = CCret; %choose char conc of RA
a = 1/k5; %choose any param that is 1/time, using k5 from wnt signaling pathway

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Linking Pathways: WNT on RAS (incr. CYP synthesis)
%bctf = @(x) TCF0.*x./(K16+w*x); %formula for conc. of bcat/tcf at any time %MAKE SURE NONDIM!
% bctf = @(x, Ci) TCF0 .* x ./ (K16_max * (1 + ((w*Ci).^nc) ./ (Cs.^nc + (w*Ci).^nc)) + w * x);
bctf = @(x, Ci) TCF0 .* x ./ (K16_0 + (K16_max .* ((w*Ci).^nc)) ./ (Cs.^nc + ((w*Ci).^nc)) + w * x);
%cyp_synthFunc = @ (x,y) (rv18 + wntef*bctf(y) + MCsynth*(r*x).^2)./(1+(r*x).^2); %is written as a function of (RA, B-catenin)
cyp_synthFunc = @(x,y,Ci) (rv18 + wntef*bctf(y, Ci) + MCsynth*(r*x).^2)./(1+(r*x).^2);


% Define a linked function to get RA link on WNT to match expected results.
%This new link is through APC degradation instead synth as before.
Pmin = 0.002;
APCdeg = @(x, y) (k19n./(1 +retef*r*x.^2)).*(y>=0.75*Pmin) + (k19n.*(1+retef.*r*x.^2)).*(y<0.75*Pmin);
%K16n = (w*K16)/(1+retef*RSS);
gamma = 0.025; %gamma = 0.005;%0.05 % original 0.025

 %b= 150;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Sources for Both Pathways
%W = @(t) 1.*(t>0); %Wnt always on
W = @(t) 0.*(t<3000).*(t>25000) + 3*1.*(t>=3000).*(t<=25000); %wnt off




% %Periodic source for RAS pathway
A = 1e2;    %amplitude
B = pi/6;   %controls period pi/6 = 12hrs
C = 0;      %controls phase shift
D = 2e2;
%gtild = @(t) (a/r)*(A + D*cos(B*a*t - C));
gtild = @(t)(a/r)*A*(1 + cos(B*a*t - C));
% %ftild = @(t) 0.1; %Bs = pi/100; %gtild_s = @(t)(a/r)*A*(1 + cos(Bs*a*t - C));
% % 
% %     % A = 0.0753*1e1;
% %     % B = 0.03*1e1;
%     A = 600;
%     B = 400;
%     C = pi/6;
%     T = 1; % Period in hours
%     phi = 0; % Phase shift
%     gtild = @(t) (a/r)*(A + B * sin((C* a * t) + phi));



% Parameters for the Generalized Gaussian function
% A = 1;        % Peak amplitude
% mu = 4500;      % Center of the peak (time of maximum effect)
% sigma = 200;    % Width of the bell
% p = 2;        % Shape parameter (2 for standard Gaussian)
% 
% % Define the Generalized Gaussian function
% f = @(t) (a/r) * A * exp(-((abs(t - mu) / sigma).^p));
% treatvals = f(time);

% % Parameters for the logistic-based bell-shaped function
% L1 = 1.5;        % Amplitude of the rising phase
% L2 = 2;        % Amplitude of the falling phase
% t1 = 3000;       % Time when rise begins
% t2 = 4000;       % Time when fall begins
% k1 = 0.1;      % Growth rate for the rising phase
% k2 = 0.2;      % Decay rate for the falling phase
% 
% 
% % Define the function
% f = @(t) (a/r) * (L1 ./ (1 + exp(-k1 * (t - t1)))) - (L2 ./ (1 + exp(-k2 * (t - t2))));
% 
% treatvals = f(time);

% s = 1e3;
% c = 100;
% treat = @ (t) (a/r).*s.*exp((-(t-4500).^2)./(2*c^2));
% treatvals = treat(time);


% treatvals = treat(time);

t_start = 10000;         % Treatment starts
t_end = 15000;           % Treatment ends
dose_strength = 0e2;    % Scaled-up dose strength 1.2*2.5e2
q = 100;                % Smoothness factor 100-300

% Smooth treatment function using tanh
treat = @(t) (a/r) * (dose_strength / 2) * ...
             (tanh((t - t_start) / q) - tanh((t - t_end) / q));

treatvals = treat(time);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Define System
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define the explicit differential equations for WNT:
dVdtn  = @(t,V,Di,Db,Bp,Da,P,Ba,X) k1n*(dsh0-V)*W(t)-k2n*V;

dDidtn = @(t,V,Di,Db,Bp,Da,P,Ba,X) -(k3n*V+k4n+k_6n).*Di+Da+(k6n.*P.*X.*...
                                   (gsk0-(1+K8n*Ba).*Da-Di-Db))./(K21+w*X);
                               
dDbdtn = @(t,V,Di,Db,Bp,Da,P,Ba,X) k9n*Da.*Ba-k10n*Db;

dBpdtn = @(t,V,Di,Db,Bp,Da,P,Ba,X) k10n*Db-k11n*Bp;


% Define the implicit differential equations:
An = @(t,V,Di,Db,Bp,Da,P,Ba,X) -K8n*(K8+w*Ba).*X./(K21+w*X);
Bn = @(t,V,Di,Db,Bp,Da,P,Ba,X) K7n*X;
Cn = @(t,V,Di,Db,Bp,Da,P,Ba,X) K20n*X-w*K8n*Da.*X./(K21+w*X);
Dn = @(t,V,Di,Db,Bp,Da,P,Ba,X) 1+K7n*P+K20n*Ba+K21*w*...
                               (gsk0-(1+K8n*Ba).*Da-Di-Db)./((K21+w*X).^2);
En = @(t,V,Di,Db,Bp,Da,P,Ba,X) (1+K8n*Ba);
Fn = @(t,V,Di,Db,Bp,Da,P,Ba,X) K8n*Da;
Gn = @(t,V,Di,Db,Bp,Da,P,Ba,X) K8n*Ba;
Hn = @(t,V,Di,Db,Bp,Da,P,Ba,X) K17n*Ba;
%In = @(t,V,Di,Db,Bp,Da,P,Ba,X) 1+K8n*Da+K16n*tcf0./((K16+w*Ba).^2)+K17n*P+K20n*X;
In = @(t,V,Di,Db,Bp,Da,P,Ba,X,Ci) ...
     1 + K8n * Da + ((K16_0 + (K16_max .* ((w*Ci).^nc)) ./ (Cs.^nc + ((w*Ci).^nc))) .* tcf0) ./ ((K16_0 + (K16_max .* ((w*Ci).^nc)) ./ (Cs.^nc + ((w*Ci).^nc)) + w * Ba).^2) + K17n * P + K20n * X;

Jn = @(t,V,Di,Db,Bp,Da,P,Ba,X) K20n*Ba;
Kn = @(t,V,Di,Db,Bp,Da,P,Ba,X) 1+K8n*Ba;
Ln = @(t,V,Di,Db,Bp,Da,P,Ba,X) 1+K7n*X+K17n*Ba;
Mn = @(t,V,Di,Db,Bp,Da,P,Ba,X) K8n*Da+K17n*P;
Nn = @(t,V,Di,Db,Bp,Da,P,Ba,X) K7n*P;
                           
                           

RHS1n = @(t,V,Di,Db,Bp,Da,P,Ba,X,Ci) v14n+(k3n*V+k_6n).*Di-...
                                  k6n.*P.*X.*...
                                  (gsk0-(1+K8n*Ba).*Da-Di-Db)./(K21+w*X)-...
                                  k15n*P.*X./(Km+w*P)+w*X./(K21+w*X).*...
                                  (dDidtn(t,V,Di,Db,Bp,Da,P,Ba,X)+...
                                  dDbdtn(t,V,Di,Db,Bp,Da,P,Ba,X));
                              
RHS2n = @(t,V,Di,Db,Bp,Da,P,Ba,X) k4n*Di-(1+k9n*Ba).*Da+k10n*Db;

%RHS3n = @(t,V,Di,Db,Bp,Da,P,Ba,X) v12n-(k13n+k9n*Da).*Ba; No change made
RHS3n =  @(t,V,Di,Db,Bp,Da,P,Ba,X,H5,H13,Ci) v12n - (k13n + k9n*Da + kp40n*H13 + ...
    k41n*H5).*Ba + km40n*Ci; % Ba further reduced by HOX proteins



% RHS4n = @(t,V,Di,Db,Bp,Da,P,Ba,X,R,H13,Ci) v18n./(1+Ktn*Ba./((30 + ((K16_max - 30)*w*Ci) / (w*Ci + Cs))+w*Ba)+Kbn*Ba + gamma*w*H13)-... %addition here for RA and H13 feedback on APC synth
%                                   APCdeg(R, P)*P-...
%                                   (dDidtn(t,V,Di,Db,Bp,Da,P,Ba,X)+...
%                                   dDbdtn(t,V,Di,Db,Bp,Da,P,Ba,X));

RHS4n = @(t,V,Di,Db,Bp,Da,P,Ba,X,R,H13,Ci) v18n./(1+Ktn*Ba./((K16_0 + (K16_max .* ((w*Ci).^nc)) ./ (Cs.^nc + ((w*Ci).^nc)))+w*Ba)+Kbn*Ba + gamma*w*H13)-... %addition here for RA and H13 feedback on APC synth
                                  APCdeg(R, P)*P-...
                                  (dDidtn(t,V,Di,Db,Bp,Da,P,Ba,X)+...
                                  dDbdtn(t,V,Di,Db,Bp,Da,P,Ba,X));




% RHS4n = @(t,V,Di,Db,Bp,Da,P,Ba,X,R,H5,H13) b*H5*v18n./(1+Ktn*Ba./(K16+w*Ba)+Kbn*Ba + gamma*w*H13)-... %addition here for RA and H13 feedback on APC synth
%                                   APCdeg(R, P)*P-...
%                                   (dDidtn(t,V,Di,Db,Bp,Da,P,Ba,X)+...
%                                   dDbdtn(t,V,Di,Db,Bp,Da,P,Ba,X));






% Define the Matrix and Vector:
Matrixn = @(t,V,Di,Db,Bp,Da,P,Ba,X,Ci) [An(t,V,Di,Db,Bp,Da,P,Ba,X), ...
                                     Bn(t,V,Di,Db,Bp,Da,P,Ba,X), ...
                                     Cn(t,V,Di,Db,Bp,Da,P,Ba,X), ...
                                     Dn(t,V,Di,Db,Bp,Da,P,Ba,X);
                                     En(t,V,Di,Db,Bp,Da,P,Ba,X), ...
                                     0, ...                                      
                                     Fn(t,V,Di,Db,Bp,Da,P,Ba,X), ...
                                     0;                                      
                                     Gn(t,V,Di,Db,Bp,Da,P,Ba,X), ...
                                     Hn(t,V,Di,Db,Bp,Da,P,Ba,X), ...
                                     In(t,V,Di,Db,Bp,Da,P,Ba,X,Ci), ...
                                     Jn(t,V,Di,Db,Bp,Da,P,Ba,X); ...
                                     Kn(t,V,Di,Db,Bp,Da,P,Ba,X), ...
                                     Ln(t,V,Di,Db,Bp,Da,P,Ba,X), ...
                                     Mn(t,V,Di,Db,Bp,Da,P,Ba,X), ...
                                     Nn(t,V,Di,Db,Bp,Da,P,Ba,X)];

Vectorn = @(t,V,Di,Db,Bp,Da,P,Ba,X,R,H5,H13,Ci) [RHS1n(t,V,Di,Db,Bp,Da,P,Ba,X,Ci);
                                     RHS2n(t,V,Di,Db,Bp,Da,P,Ba,X);
                                     RHS3n(t,V,Di,Db,Bp,Da,P,Ba,X,H5,H13,Ci);
                                     RHS4n(t,V,Di,Db,Bp,Da,P,Ba,X,R,H13,Ci)];



% Vectorn = @(t,V,Di,Db,Bp,Da,P,Ba,X,R,H5,H13,Ci) [RHS1n(t,V,Di,Db,Bp,Da,P,Ba,X,Ci);
%                                      RHS2n(t,V,Di,Db,Bp,Da,P,Ba,X);
%                                      RHS3n(t,V,Di,Db,Bp,Da,P,Ba,X,H5,H13,Ci);
%                                      RHS4n(t,V,Di,Db,Bp,Da,P,Ba,X,R,H5,H13)];


% Solutionn = @(t,V,Di,Db,Bp,Da,P,Ba,X,R,H5,H13,Ci) Matrixn(t,V,Di,Db,Bp,Da,P,Ba,X)\...
%                                       Vectorn(t,V,Di,Db,Bp,Da,P,Ba,X,R,H5,H13,Ci);
% 
% 
% Solutionn = @(t,V,Di,Db,Bp,Da,P,Ba,X,R,H5,H13,Ci) ...
%     Matrixn(t,V,Di,Db,Bp,Da,P,Ba,X) \ ...
%     Vectorn(t,V,Di,Db,Bp,Da,P,Ba,X,R,H5,H13,Ci);

Solutionn = @(t, V, Di, Db, Bp, Da, P, Ba, X, R, H5, H13, Ci) ...
        Matrixn(t, V, Di, Db, Bp, Da, P, Ba, X, Ci) \ Vectorn(t, V, Di, Db, Bp, Da, P, Ba, X, R, H5, H13, Ci);



                                  
%Set up RAS Pathway Eqns 

dRodtn = @(t,Ro,Ra,A,R,B,Br,N,Nr,C,Cr,Dc,Dn,Bc) a*(rkm1.*Ra - rkp1.*Ro)+gtild(t);
dRadtn = @(t,Ro,Ra,A,R,B,Br,N,Nr,C,Cr,Dc,Dn,Bc) a*(rkp1.*Ro-(rkm1 + rk2*r.*A).*Ra);
dAdtn  = @(t,Ro,Ra,A,R,B,Br,N,Nr,C,Cr,Dc,Dn,Bc) (a/r)*rv19 - a*(rk23.*A);  
dRdtn  = @(t,Ro,Ra,A,R,B,Br,N,Nr,C,Cr,Dc,Dn,Bc) treat(t) + a*(rk2*r.*Ra.*A - rkp3*r.*R.*B + (rkm3+rk14).*Br - rkp4*r.*R.*N + (rkm4+rk15).*Nr -rkp5*r.*R.*C +rkm5.*Cr);
dBdtn  = @(t,Ro,Ra,A,R,B,Br,N,Nr,C,Cr,Dc,Dn,Bc) (a/r)*rv16 + a*(-(rk20 + rkp3*r.*R).*B + rkm3.*Br + rk10.*Dn);
dBrdtn = @(t,Ro,Ra,A,R,B,Br,N,Nr,C,Cr,Dc,Dn,Bc) a*(rkp3*r.*R.*B - rkm3.*Br - rkp6*r.*Br.*C + rkm6.*Dc - rkp9*r.*Br.*N + rkm9.*Dn -rk14.*Br);
dNdtn  = @(t,Ro,Ra,A,R,B,Br,N,Nr,C,Cr,Dc,Dn,Bc) (a/r)*rv17 + a*(-(rk21 + rkp4*r.*R).*N + rkm4.*Nr -rkp9*r.*Br.*N + rkm9.*Dn);
dNrdtn = @(t,Ro,Ra,A,R,B,Br,N,Nr,C,Cr,Dc,Dn,Bc) a*(rkp4*r.*R.*N - (rkm4+rk15).*Nr +rk10.*Dn);
%dCdtn  = @(t,Ro,Ra,A,R,B,Br,N,Nr,C,Cr,Dc,Dn,Bc,Ba) (a/r)*cyp_synthFunc(R,Ba)+ a*(-(rk22+rkp5*r.*R).*C + (rkm5+rk8).*Cr - rkp6*r.*Br.*C + rkm6.*Dc);
dCdtn  = @(t,Ro,Ra,A,R,B,Br,N,Nr,C,Cr,Dc,Dn,Bc,Ba,Ci) ...
         (a/r)*cyp_synthFunc(R,Ba,Ci) + a*(-(rk22+rkp5*r.*R).*C + (rkm5+rk8).*Cr - rkp6*r.*Br.*C + rkm6.*Dc);
dCrdtn = @(t,Ro,Ra,A,R,B,Br,N,Nr,C,Cr,Dc,Dn,Bc) a*(rkp5*r.*R.*C - (rkm5+rk8).*Cr);
dDcdtn = @(t,Ro,Ra,A,R,B,Br,N,Nr,C,Cr,Dc,Dn,Bc) a*(rkp6*r.*Br.*C - (rkm6+rk7).*Dc);
dDndtn = @(t,Ro,Ra,A,R,B,Br,N,Nr,C,Cr,Dc,Dn,Bc) a*(rkp9*r.*Br.*N - (rkm9+rk10).*Dn);
dBcdtn = @(t,Ro,Ra,A,R,B,Br,N,Nr,C,Cr,Dc,Dn,Bc) a*rk7.*Dc;



%u14 = tcf0 * Ba ./ (1 + Ba); % Bcat/TCF replaced by bctf
%k33 = @(Nr) (k33_max *(w*Nr).^n)/(Kd^n + (w*Nr).^n); % new synth of HOXA13
%k33n = @(Nr) (Kd+a1*(w*Nr).^n)/(1 + (w*Nr).^n);
k35n = @(Ca) (k34 + a3 * (w * Ca).^n3) / (1 + (phi*w * Ca).^n3) / k5; % Deg of HOXA5 by MYC/MIZ1 complex
v30n = @(bctf) (Kc + a2 * w * bctf) / (1 + w * bctf) / (w * k5); % MYC synth
%v32n = @(Ba) (Kd + a4 * K16 * Ba)/(1 + K16 * Ba); % new synth of HOXA13




%dH5dtn  = @(t,H5,M,Mi,Ca,H13,Ci,Nr,Ba) knn*((w*Mi).^n1/(a5.^n1 +(w*Mi).^n1)).*((w*Nr).^n2/(a6.^n2 +(w*Nr).^n2)) - k35n(Ca).*H5;

%dH5dtn  = @(t,H5,M,Mi,Ca,H13,Ci,Nr,Ba) knn*((w*Mi).^n1/(a5.^n1 +(w*Mi).^n1)).*((w*Nr).^n2/(a6.^n2 +(w*Nr).^n2)) - k35n*(k34 + a3*w*Ca).*H5;
%dH5dtn  = @(t,H5,M,Mi,Ca,H13,Ci,Nr,Ba) k31n + k32n*Mi + k33n*Nr - k35n*(k34 + a3*w*Ca).*H5;

% Differential equations of the HOX model
%dH5dtn  = @(t,H5,M,Mi,Ca,H13,Ci,Nr,Ba) k31n + (k32n*(w*Mi).^n1)/(a5.^n1 +(w*Mi).^n1) + (k33n*(w*Nr).^n2)/(a6.^n2 +(w*Nr).^n2) - k35n(Ca).*H5;
dH5dtn  = @(t,H5,M,Mi,Ca,H13,Ci,Nr,Ba) k31n + k32n*Mi + k33n(Nr).*Nr - k35n(Ca).*H5;
%dH5dtn  = @(t,H5,M,Mi,Ca,H13,Ci,Nr,Ba) k31n + k32n*Mi + k33n*Nr - k35n(Ca).*H5;
%dMdtn   = @(t,H5,M,Mi,Ca,H13,Ci,Nr,Ba) v30n(bctf(Ba)) - k36n.* M - kp37n.* M.*Mi + km37n.*Ca;
dMdtn   = @(t,H5,M,Mi,Ca,H13,Ci,Nr,Ba) v30n(bctf(Ba, Ci)) - k36n.* M - kp37n.* M.*Mi + km37n.*Ca;
dMidtn  = @(t,H5,M,Mi,Ca,H13,Ci,Nr,Ba) v31n - k38n.*Mi - kp37n.*M.*Mi + km37n.*Ca;
dCadt   = @(t,H5,M,Mi,Ca,H13,Ci,Nr,Ba) kp37n.*M.*Mi - km37n.*Ca;
dH13dtn = @(t,H5,M,Mi,Ca,H13,Ci,Nr,Ba) v32n - k39n.*H13 - kp40n.*H13.*Ba + km40n.*Ci;
%dH13dtn = @(t,H5,M,Mi,Ca,H13,Ci,Nr,Ba) v32n(Ba) - k39n.*H13 - kp40n.*H13.*Ba + km40n.*Ci;
dCidtn  = @(t,H5,M,Mi,Ca,H13,Ci,Nr,Ba) kp40n.*H13.*Ba - km40n.*Ci;







RHSn = @(t, U) [
        dRodtn(t, U(1), U(2), U(3), U(4), U(5), U(6), U(7), U(8), U(9), U(10), U(11), U(12), U(13));
        dRadtn(t, U(1), U(2), U(3), U(4), U(5), U(6), U(7), U(8), U(9), U(10), U(11), U(12), U(13));
        dAdtn(t, U(1), U(2), U(3), U(4), U(5), U(6), U(7), U(8), U(9), U(10), U(11), U(12), U(13));
        dRdtn(t, U(1), U(2), U(3), U(4), U(5), U(6), U(7), U(8), U(9), U(10), U(11), U(12), U(13));
        dBdtn(t, U(1), U(2), U(3), U(4), U(5), U(6), U(7), U(8), U(9), U(10), U(11), U(12), U(13));
        dBrdtn(t, U(1), U(2), U(3), U(4), U(5), U(6), U(7), U(8), U(9), U(10), U(11), U(12), U(13));
        dNdtn(t, U(1), U(2), U(3), U(4), U(5), U(6), U(7), U(8), U(9), U(10), U(11), U(12), U(13));
        dNrdtn(t, U(1), U(2), U(3), U(4), U(5), U(6), U(7), U(8), U(9), U(10), U(11), U(12), U(13));
        dCdtn(t, U(1), U(2), U(3), U(4), U(5), U(6), U(7), U(8), U(9), U(10), U(11), U(12), U(13), U(20), U(27));
        dCrdtn(t, U(1), U(2), U(3), U(4), U(5), U(6), U(7), U(8), U(9), U(10), U(11), U(12), U(13));
        dDcdtn(t, U(1), U(2), U(3), U(4), U(5), U(6), U(7), U(8), U(9), U(10), U(11), U(12), U(13));
        dDndtn(t, U(1), U(2), U(3), U(4), U(5), U(6), U(7), U(8), U(9), U(10), U(11), U(12), U(13));
        dBcdtn(t, U(1), U(2), U(3), U(4), U(5), U(6), U(7), U(8), U(9), U(10), U(11), U(12), U(13));
        dVdtn(t, U(14), U(15), U(16), U(17), U(18), U(19), U(20), U(21));
        dDidtn(t, U(14), U(15), U(16), U(17), U(18), U(19), U(20), U(21));
        dDbdtn(t, U(14), U(15), U(16), U(17), U(18), U(19), U(20), U(21));
        dBpdtn(t, U(14), U(15), U(16), U(17), U(18), U(19), U(20), U(21));
        Solutionn(t, U(14), U(15), U(16), U(17), U(18), U(19), U(20), U(21), U(4), U(22), U(26), U(27));
        dH5dtn(t, U(22), U(23), U(24), U(25), U(26), U(27), U(8), U(20));
        dMdtn(t, U(22), U(23), U(24), U(25), U(26), U(27), U(8), U(20));
        dMidtn(t, U(22), U(23), U(24), U(25), U(26), U(27), U(8), U(20));
        dCadt(t, U(22), U(23), U(24), U(25), U(26), U(27), U(8), U(20));
        dH13dtn(t, U(22), U(23), U(24), U(25), U(26), U(27), U(8), U(20));
        dCidtn(t, U(22), U(23), U(24), U(25), U(26), U(27), U(8), U(20))
    ]; % Ensure Column Vector

%Solutionn(t,U(14),U(15),U(16),U(17),U(18),U(19),U(20),U(21),U(4));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initial Conditions from Katie:
V_0  = 0;
Di_0 = 4.83e-3;
Db_0 = 2.02e-3;
Bp_0 = 1;
Da_0 = 9.66e-3;
P_0  = 18.116;
%P_0 = APC0/K17; %allow decrease or increase in APC IC
Ba_0 = 25.1;
X_0  = 4.93e-4;




W_0n = (1/w)*[V_0; Di_0; Db_0; Bp_0; Da_0; P_0; Ba_0; X_0];

% Initial conditions: 
x1_0  = 10;
x2_0  = 10;
x3_0  = 10;
x4_0  = 100;%RA
x5_0  = 1;
x6_0  = 0;
x7_0  = 1;
x8_0  = 0;
x9_0  = 0.01;
x10_0 = 0;
x11_0 = 0;
x12_0 = 0;
x13_0 = 0;

R_0n = (1/1000)*[x1_0; x2_0; x3_0; x4_0; x5_0; x6_0; x7_0; x8_0; x9_0; x10_0;
       x11_0; x12_0; x13_0];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%Main Test For w =1000 
%H5_0 = 0.00003;%0.014;%0.0115; % 0.00865, 0.00835, 10.0; 26.0; 130.0, 128.0729, 130.0729
%H5_0 = 0.0033;
H5_0 = 0.00146;
%M_0 = 0.050; %1.5227; 1.5 0.045
%M_0 = 0.055;
M_0 = 0.04999;
%Mi_0 = 0.057; % 0.049, 0.055
%Mi_0 = 0.058;
Mi_0 = 0.0601;
%Ca_0 = 2.25; %5.7101, 5.0; 0.0
%Ca_0 = 0.8984; % 0.70 0.75
Ca_0 = 0.84028;
H13_0 = 0.04439 ; % 0.0333 0.0439
Ci_0 = 0.0012; %1.8901; 0




H_0n = [H5_0; M_0; Mi_0; Ca_0; H13_0; Ci_0];

U_0n = [R_0n; W_0n; H_0n];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solver:
[t,U] = ode15s(RHSn,time,U_0n);  
% BcatTcf = bctf(U(:,20));
% Sol = [U BcatTcf];
BcatTcf = bctf(U(:, 20), U(:, 27));

Sol = [U, BcatTcf];

                 


end