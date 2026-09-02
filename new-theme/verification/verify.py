import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
# Part A params
r1,r2,r3=1.0,0.8,0.8; K1,K2,K3=1.0,2.0,2.0
a12=a13=0.89; h12=h13=1.0; a21=a31=0.55; h21=h31=0.5; a23=a32=0.5; h23=h32=1.0; gam=1.0
def f(t,y,beta):
    u,v,w=y
    Ku=K1+beta*u/(1+gam*u)
    du=r1*u*(1-u/Ku)-u*(a12*v/(1+h12*v)+a13*w/(1+h13*w))
    dv=r2*v*(1-v/K2)-v*(a21*u/(1+h21*u)+a23*w/(1+h23*w))
    dw=r3*w*(1-w/K3)-w*(a31*u/(1+h31*u)+a32*v/(1+h32*v))
    return [du,dv,dw]
def jac(y,beta,eps=1e-6):
    J=np.zeros((3,3)); y=np.array(y,float)
    for j in range(3):
        e=np.zeros(3); e[j]=eps
        J[:,j]=(np.array(f(0,y+e,beta))-np.array(f(0,y-e,beta)))/(2*eps)
    return J
# V0
V0=max(np.roots([-r2*h23/K2, r2*h23-r2/K2-a23, r2])); print("V0",V0)
lam=r1-2*a12*V0/(1+h12*V0); print("lambda_invade",lam)
print("beta2 closed",247/48, "u*",r2/(a21-r2*h21))
# regimes: long integration from engineer-rich IC
for beta in [2.0,3.7,3.8,4.0,4.2,4.3,4.8,5.2,5.6]:
    s=solve_ivp(f,[0,4000],[3.0,0.22,0.18],args=(beta,),rtol=1e-9,atol=1e-12)
    y=s.y[:,-1]; print(f"beta={beta}: final {np.round(y,4)}  eig {np.round(np.linalg.eigvals(jac(y,beta)),4)}")
# invasion from rare
for beta in [0,3,6]:
    s=solve_ivp(f,[0,4000],[0.02,1,1],args=(beta,),rtol=1e-9,atol=1e-12); print("rare",beta,np.round(s.y[:,-1],4))
