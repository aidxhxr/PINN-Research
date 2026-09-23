# Converted from PINN_forward.ipynb (repo path: notebooks/PINN_forward.ipynb) on 2026-09-23.
# Cells appear in notebook order; code is verbatim. Lines that were IPython-only
# ("!shell", "%magic") are kept but commented with "[ipython-only, skipped]".
# Run from this directory:  cd notebooks && python3 PINN_forward.py

# %% [cell 1]
# Assumptions for normasysl_call.m:
# constant-K16 (gamma1=0, black)  vs  dynamic-K16 (gamma1=1, red),
# WNT-on region [3000,25000] shaded, axes identical (x: time tau, y: non-dim).
import numpy as np,torch, torch.nn as nn, matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [cell 2]
# state layout U(k)_matlab in matlab into z[:,k-1] 
iRo,iRa,iA,iR,iB,iBr,iN,iNr,iC,iCr,iDc,iDn,iBc = range(0,13)
iV,iDi,iDb,iBp,iDa,iP,iBa,iX = range(13,21)
iH5,iM,iMi,iCa,iH13,iCi = range(21,27)

# %% [cell 3]
# Setting up parameters from their definition in matlab. gamma1 toggles dynamic vs constant K16.
def build_params(gamma1=1.0, funcpercent=1.0, retef=0.0015, wntef=100.0,
                 CCret=1000.0, CCwnt=1000.0):
    P={}; w=CCwnt; r=CCret; k5=0.133; a=1.0/k5
    w, thetaP, nb, nM, nH=0.8, 1.0, 2, 2, 2
    
    P.update(w=w,r=r,k5=k5,a=a,retef=retef,wntef=wntef)
    DSH0,TCF0,GSK0=100.,15.,50.; K8,K17=120.,1200.
    K16_0,K16_max,Cs,nc=100.,200.*gamma1,0.005,2
    K7,K20,K21,Km=50.,1.,1.,98.
    k1,k2,k3,k4=0.182,1.82e-2,5e-2,0.267
    k6,k_6,k9,k10,k11=9.09e-2,0.909,206.,206.,0.417
    v12,k13,v14,k15=0.423,2.57e-4,8.22e-5,0.33
    Pars=[212.8453,39.9102,34.1111]; k19=1.0/K17
    v18=Pars[0]*k19; Kt,Kb=Pars[1],Pars[2]
    if funcpercent<1: v18*=funcpercent
    P.update(gsk0=GSK0/w,tcf0=TCF0/w,dsh0=DSH0/w,TCF0=TCF0,
             K16_0=K16_0,K16_max=K16_max,Cs=Cs,nc=nc,K21=K21,Km=Km,
             K7n=w/K7,K8n=w/K8,K8=K8,K17n=w/K17,K20n=w/K20,
             Ktn=(TCF0*w)/Kt,Kbn=w/Kb,
             k1n=k1/k5,k2n=k2/k5,k3n=k3*w/k5,k4n=k4/k5,
             k6n=(k6*K21*w**2)/(k5*K7),k_6n=k_6/k5,
             k9n=(k9*w)/(k5*K8),k10n=k10/k5,k11n=k11/k5,
             v12n=v12/(w*k5),k13n=k13/k5,v14n=v14/(k5*w),
             k15n=k15*w/k5,v18n=v18/(w*k5),k19n=k19/k5,gamma=0.025)
    p=[0.5,0.1,0.5,(2.13e11)*1e-9,1.32e1,(6.00e9)*1e-0,3.60e1,(1.2e11)*1e-9,
       4.00e1,1.01e1,1.01e1,1.02e1,2.70e1,(3.60e9)*1e-7,3.00e1,1.00e-1,
       3.85e-2,1.73e-3,2.00e-1,2.00e-1,2.00e-2,2.50e-2,3.85e-2,1.73e-1,
       8.35e-3,1.00e-2,1.00e-1]
    (P['rkp1'],P['rkm1'],P['rk2'],P['rkp3'],P['rkm3'],P['rkp4'],P['rkm4'],
     P['rkp5'],P['rkm5'],P['rkp6'],P['rkm6'],P['rk7'],P['rk8'],P['rkp9'],
     P['rkm9'],P['rk10'],P['rk14'],P['rk15'],P['rv16'],P['rv17'],P['rv18'],
     P['rv19'],P['rk20'],P['rk21'],P['rk22'],P['rk23'],P['MCsynth'])=p
    k31=8.25*16.8182e2; k32=20.85*1.25e1; k34=0.25*10; k36=0.13
    kp37=0.45*1e-1; km37=0.1*1.65; v31=0.15*3.333e1; k38=0.083
    v32=0.20*3.333e1; k39=0.15
    P.update(k31n=k31/(w*k5),k32n=k32/k5,k36n=k36/k5,kp37n=kp37*w/k5,
             km37n=km37/k5,v31n=v31/(w*k5),k38n=k38/k5,v32n=v32/(w*k5),
             k39n=k39/k5,kp40n=(2.5e-6*w)/k5,km40n=1.0e-3/k5,k41n=(0.1475e-4*w)/k5,
             a2=7.735,a3=50.5,Kc=1.0,Kd=0.01,k33_max=1.0,nexp=1,n3=1,
             phi=0.00139,k34=k34)
    return P

# %% [cell 4]
# forcing same t variable as ode15s, t in [0,30000]. It should work better because SciPy doesn't have ode15s
def W(t,P):     return 3.0*((t>=3000.)&(t<=25000.)).to(t.dtype)
def gtild(t,P): return (P['a']/P['r'])*1e2*(1.0+torch.cos((np.pi/6)*P['a']*t))
def treat(t,P): return torch.zeros_like(t)

# %% [cell 5]
# couplings
def K16_dyn(Ci,P):
    wCi=P['w']*Ci
    return P['K16_0']+P['K16_max']*(wCi**P['nc'])/(P['Cs']**P['nc']+wCi**P['nc'])
def bctf(Ba,Ci,P):       return P['TCF0']*Ba/(K16_dyn(Ci,P)+P['w']*Ba)
def cyp_synth(Rv,Ba,Ci,P):
    rR2=(P['r']*Rv)**2
    return (P['rv18']+P['wntef']*bctf(Ba,Ci,P)+P['MCsynth']*rR2)/(1.0+rR2)
def APCdeg(Rv,Pv,P):
    Pmin=0.002; hi=(Pv>=0.75*Pmin).to(Rv.dtype)
    return (P['k19n']/(1.0+P['retef']*P['r']*Rv**2))*hi \
         + (P['k19n']*(1.0+P['retef']*P['r']*Rv**2))*(1.0-hi)


# %% [cell 6]
# initial state to start the function. They are non-dimensional lol
def initial_state(P):
    w=P['w']
    RA=np.array([10,10,10,100,1,0,1,0,0.01,0,0,0,0])/1000.0
    WN=np.array([0,4.83e-3,2.02e-3,1,9.66e-3,18.116,25.1,4.93e-4])/w
    HX=np.array([0.00146,0.04999,0.0601,0.84028,0.04439,0.0012])
    return np.concatenate([RA,WN,HX])

# %% [cell 7]
# defining all of the residual errors. On this, I used Claude Code because there are 27 residual errors.
# it's part of the loss function. But it's the physical loss

def residuals(t,z,dz,P):
    w,r,a=P['w'],P['r'],P['a']; g=lambda i:z[:,i:i+1]
    Ro,Ra,A,Rv,Bx,Br,Nx,Nr,Cx,Cr,Dc,Dn,Bc=[g(i) for i in range(0,13)]
    V,Di,Db,Bp,Da,Pv,Ba,X=[g(i) for i in range(13,21)]
    H5,M,Mi,Ca,H13,Ci=[g(i) for i in range(21,27)]
    # RA
    fRo=a*(P['rkm1']*Ra-P['rkp1']*Ro)+gtild(t,P)
    fRa=a*(P['rkp1']*Ro-(P['rkm1']+P['rk2']*r*A)*Ra)
    fA =(a/r)*P['rv19']-a*P['rk23']*A
    fR =treat(t,P)+a*(P['rk2']*r*Ra*A-P['rkp3']*r*Rv*Bx+(P['rkm3']+P['rk14'])*Br
        -P['rkp4']*r*Rv*Nx+(P['rkm4']+P['rk15'])*Nr-P['rkp5']*r*Rv*Cx+P['rkm5']*Cr)
    fB =(a/r)*P['rv16']+a*(-(P['rk20']+P['rkp3']*r*Rv)*Bx+P['rkm3']*Br+P['rk10']*Dn)
    fBr=a*(P['rkp3']*r*Rv*Bx-P['rkm3']*Br-P['rkp6']*r*Br*Cx+P['rkm6']*Dc
        -P['rkp9']*r*Br*Nx+P['rkm9']*Dn-P['rk14']*Br)
    fN =(a/r)*P['rv17']+a*(-(P['rk21']+P['rkp4']*r*Rv)*Nx+P['rkm4']*Nr
        -P['rkp9']*r*Br*Nx+P['rkm9']*Dn)
    fNr=a*(P['rkp4']*r*Rv*Nx-(P['rkm4']+P['rk15'])*Nr+P['rk10']*Dn)
    fC =(a/r)*cyp_synth(Rv,Ba,Ci,P)+a*(-(P['rk22']+P['rkp5']*r*Rv)*Cx
        +(P['rkm5']+P['rk8'])*Cr-P['rkp6']*r*Br*Cx+P['rkm6']*Dc)
    fCr=a*(P['rkp5']*r*Rv*Cx-(P['rkm5']+P['rk8'])*Cr)
    fDc=a*(P['rkp6']*r*Br*Cx-(P['rkm6']+P['rk7'])*Dc)
    fDn=a*(P['rkp9']*r*Br*Nx-(P['rkm9']+P['rk10'])*Dn)
    fBc=a*P['rk7']*Dc
    # WNT explicit
    fV=P['k1n']*(P['dsh0']-V)*W(t,P)-P['k2n']*V
    pool=(P['gsk0']-(1+P['K8n']*Ba)*Da-Di-Db)
    fDi=-(P['k3n']*V+P['k4n']+P['k_6n'])*Di+Da+(P['k6n']*Pv*X*pool)/(P['K21']+w*X)
    fDb=P['k9n']*Da*Ba-P['k10n']*Db
    fBp=P['k10n']*Db-P['k11n']*Bp
    # HOX
    k33n=((P['k33_max']*(w*Nr)**P['nexp'])/(P['Kd']**P['nexp']+(w*Nr)**P['nexp']))/P['k5']
    k35n=(P['k34']+P['a3']*(w*Ca)**P['n3'])/(1+(P['phi']*w*Ca)**P['n3'])/P['k5']
    bt=bctf(Ba,Ci,P); v30n=(P['Kc']+P['a2']*w*bt)/(1+w*bt)/(w*P['k5'])
    fH5=P['k31n']+P['k32n']*Mi+k33n*Nr-k35n*H5
    fM =v30n-P['k36n']*M-P['kp37n']*M*Mi+P['km37n']*Ca
    fMi=P['v31n']-P['k38n']*Mi-P['kp37n']*M*Mi+P['km37n']*Ca
    fCa=P['kp37n']*M*Mi-P['km37n']*Ca
    fH13=P['v32n']-P['k39n']*H13-P['kp40n']*H13*Ba+P['km40n']*Ci
    fCi=P['kp40n']*H13*Ba-P['km40n']*Ci
    f=torch.cat([fRo,fRa,fA,fR,fB,fBr,fN,fNr,fC,fCr,fDc,fDn,fBc,
                 fV,fDi,fDb,fBp,torch.zeros_like(Da),torch.zeros_like(Pv),
                 torch.zeros_like(Ba),torch.zeros_like(X),fH5,fM,fMi,fCa,fH13,fCi],1)
    expl=[i for i in range(27) if i not in (iDa,iP,iBa,iX)]
    res_expl=dz[:,expl]-f[:,expl]
    # WNT implicit mass-matrix
    K8n,K7n,K17n,K20n,K21=P['K8n'],P['K7n'],P['K17n'],P['K20n'],P['K21']
    An_=-K8n*(P['K8']+w*Ba)*X/(K21+w*X); Bn_=K7n*X
    Cn_=K20n*X-w*K8n*Da*X/(K21+w*X)
    Dn_=1+K7n*Pv+K20n*Ba+K21*w*pool/((K21+w*X)**2)
    En_=1+K8n*Ba; Fn_=K8n*Da; Gn_=K8n*Ba; Hn_=K17n*Ba
    K16e=K16_dyn(Ci,P)
    In_=1+K8n*Da+(K16e*P['tcf0'])/((K16e+w*Ba)**2)+K17n*Pv+K20n*X
    Jn_=K20n*Ba; Kn_=1+K8n*Ba; Ln_=1+K7n*X+K17n*Ba; Mn_=K8n*Da+K17n*Pv; Nn_=K7n*Pv
    RHS1=P['v14n']+(P['k3n']*V+P['k_6n'])*Di-P['k6n']*Pv*X*pool/(K21+w*X) \
         -P['k15n']*Pv*X/(P['Km']+w*Pv)+w*X/(K21+w*X)*(fDi+fDb)
    RHS2=P['k4n']*Di-(1+P['k9n']*Ba)*Da+P['k10n']*Db
    RHS3=P['v12n']-(P['k13n']+P['k9n']*Da+P['kp40n']*H13+P['k41n']*H5)*Ba+P['km40n']*Ci
    dK=(P['K16_0']+P['K16_max']*((w*Ci)**P['nc'])/(P['Cs']**P['nc']+(w*Ci)**P['nc']))
    RHS4=P['v18n']/(1+P['Ktn']*Ba/(dK+w*Ba)+P['Kbn']*Ba+P['gamma']*w*H13) \
         -APCdeg(Rv,Pv,P)*Pv-(fDi+fDb)
    dDa,dP,dBa,dX=dz[:,iDa:iDa+1],dz[:,iP:iP+1],dz[:,iBa:iBa+1],dz[:,iX:iX+1]
    r1=An_*dDa+Bn_*dP+Cn_*dBa+Dn_*dX-RHS1
    r2=En_*dDa+Fn_*dBa-RHS2
    r3=Gn_*dDa+Hn_*dP+In_*dBa+Jn_*dX-RHS3
    r4=Kn_*dDa+Ln_*dP+Mn_*dBa+Nn_*dX-RHS4
    return res_expl, torch.cat([r1,r2,r3,r4],1)

# %% [cell 8]
# Network class. Taken from the BINN papers
class PINN(nn.Module):
    def __init__(self,n,scale,t_win,P,nodes=128,layers=5):
        super().__init__()
        self.register_buffer("scale",torch.as_tensor(scale).reshape(1,-1))
        self.t_win=t_win
        Bf=(np.pi/6)*P['a']; self.register_buffer("freqs",torch.tensor([Bf,2*Bf]))
        seq=[nn.Linear(5,nodes),nn.Tanh()]
        for _ in range(layers-1): seq+=[nn.Linear(nodes,nodes),nn.Tanh()]
        seq+=[nn.Linear(nodes,n)]; self.seq=nn.Sequential(*seq)
        for m in self.seq:
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight,gain=0.5); nn.init.zeros_(m.bias)
    def feat(self,t,t0):
        s=(t-t0)/self.t_win; ft=t*self.freqs.reshape(1,-1)
        return torch.cat([s,torch.sin(ft),torch.cos(ft)],1)
    def forward(self,t,t0,y0):
        s=(t-t0)/self.t_win
        return y0+self.seq(self.feat(t,t0))*s*self.scale


# %% [cell 9]

def time_deriv(net,t,t0,y0):
    t=t.clone().requires_grad_(True); z=net(t,t0,y0); dz=torch.zeros_like(z)
    for i in range(z.shape[1]):
        dz[:,i:i+1]=torch.autograd.grad(z[:,i].sum(),t,create_graph=True)[0]
    return z,dz


def solve_pinn(gamma1,T=30000.,win_len=1000.,iters=3000,lr=2e-4,
               n_col=2000,n_out=300,nodes=128,layers=5,seed=0,verbose=True):
    torch.manual_seed(seed); P=build_params(gamma1=gamma1)
    y0_np=initial_state(P)
    scale=np.maximum(np.abs(y0_np),0.05)
    edges=sorted(set(list(np.arange(0.,T+win_len,win_len))+[3000.,25000.]))
    edges=np.array([e for e in edges if e<=T+1e-9])
    y0=torch.tensor(y0_np,device=device).reshape(1,-1)
    out_t,out_z=[],[]
    for w in range(len(edges)-1):
        t0,t1=float(edges[w]),float(edges[w+1])
        net=PINN(27,scale,t1-t0,P,nodes,layers).to(device)
        opt=torch.optim.Adam(net.parameters(),lr=lr)
        t0c=torch.full((n_col,1),t0,device=device)
        for it in range(iters):
            ts=torch.rand(n_col,1,device=device)*(t1-t0)+t0
            opt.zero_grad()
            z,dz=time_deriv(net,ts,t0c,y0)
            re,ri=residuals(ts,z,dz,P)
            loss=(re**2).mean()+(ri**2).mean()
            loss.backward(); opt.step()
        with torch.no_grad():
            tg=torch.linspace(t0,t1,n_out,device=device).reshape(-1,1)
            zg=net(tg,torch.full_like(tg,t0),y0)
            out_t.append(tg.cpu().numpy().ravel()); out_z.append(zg.cpu().numpy())
            y0=net(torch.full((1,1),t1,device=device),
                   torch.full((1,1),t0,device=device),y0).detach()
        if verbose:
            print(f"  [g{gamma1}] win {w+1:>3}/{len(edges)-1}  "
                  f"t<= {t1:>6.0f}  loss={loss.item():.2e}")
    t=np.concatenate(out_t); Z=np.concatenate(out_z)
    Pp=build_params(gamma1=gamma1)
    Ba=Z[:,iBa]; Ci=Z[:,iCi]
    BcatTcf=Pp['TCF0']*Ba/(K16_dyn(torch.tensor(Ci),Pp).numpy()+Pp['w']*Ba)
    return dict(t=t, H5=Z[:,iH5], H13=Z[:,iH13], M=Z[:,iM], Mi=Z[:,iMi],
                Ca=Z[:,iCa], Ci=Ci, Ba=Ba, P=Z[:,iP], Bp=Z[:,iBp], V=Z[:,iV],
                Di=Z[:,iDi], Db=Z[:,iDb], Da=Z[:,iDa], X=Z[:,iX], Nr=Z[:,iNr],
                R=Z[:,iR], BcatTcf=BcatTcf)


c_black=(0.1,0.1,0.1); c_red=(0.6,0.0,0.0); shade=(0.95,0.95,0.95)
def _panel(ax,t1,y1,t2,y2,title,shaded):
    ax.plot(t1,y1,color=c_black,lw=2.2,label='constant $K_{16}$')
    ax.plot(t2,y2,color=c_red,lw=2.2,label='dynamic $K_{16}$')
    if shaded:
        yl=ax.get_ylim()
        ax.fill_between([3000,25000],yl[0],yl[1],color=shade,zorder=0)
        ax.set_ylim(yl)
    ax.set_title(title); ax.set_xlabel(r'time ($\tau$)'); ax.set_ylabel('(non-dim)')

def plot_all(const,dyn,shaded=True):
    hox=[('H5','HOXA5 ($H_a$)'),('H13','HOXA13 ($H_i$)'),('M','MYC ($M$)'),
         ('Mi','MIZ1 ($M_i$)'),('Ca','MYC:MIZ1 ($C_a$)'),
         ('Ci',r'HOXA13:$\beta$-cat ($C_i$)')]
    fig,ax=plt.subplots(2,3,figsize=(15,9))
    for a,(k,ttl) in zip(ax.ravel(),hox):
        _panel(a,const['t'],const[k],dyn['t'],dyn[k],ttl,shaded)
    ax[0,0].legend(fontsize=9); fig.suptitle('HOX components: constant vs dynamic $K_{16}$',
        fontsize=16); fig.tight_layout()

    wnt=[('Ba',r'$\beta$-cat ($B_a$)'),('P','APC ($P$)'),
         ('BcatTcf',r'$\beta$-cat:TCF ($B_t$)'),('Bp',r'$\beta$-cat$^*$ ($B_p$)'),
         ('V',r'Dsh$_a$ ($V$)'),('Di',r'APC:Axin:GSK ($D_i$)'),
         ('Da',r'destruction$^*$ ($D_a$)'),('X','Axin ($X$)')]
    fig2,ax2=plt.subplots(2,4,figsize=(19,9))
    for a,(k,ttl) in zip(ax2.ravel(),wnt):
        _panel(a,const['t'],const[k],dyn['t'],dyn[k],ttl,shaded)
    ax2[0,0].legend(fontsize=9); fig2.suptitle('WNT components: constant vs dynamic $K_{16}$',
        fontsize=16); fig2.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------
def run_all(T=30000., win_len=1000., iters=3000):
    print("Solving CONSTANT K16 (gamma1=0)...");  const=solve_pinn(0.0,T,win_len,iters)
    print("Solving DYNAMIC  K16 (gamma1=1)...");  dyn  =solve_pinn(1.0,T,win_len,iters)
    plot_all(const,dyn);  return const,dyn

if __name__=="__main__":
    const,dyn=run_all(T=30000., win_len=200., iters=1500)

# %% [cell 10]
