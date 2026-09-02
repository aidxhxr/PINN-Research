import numpy as np
from scipy.optimize import fsolve, brentq
def make(r,K,a,h,gam):
    def f(y,beta):
        u,v,w=y; Ku=K[0]+beta*u/(1+gam*u)
        c=lambda i,j,N: a[i][j]*N/(1+h[i][j]*N)
        return np.array([r[0]*u*(1-u/Ku)-u*(c(0,1,v)+c(0,2,w)),
                         r[1]*v*(1-v/K[1])-v*(c(1,0,u)+c(1,2,w)),
                         r[2]*w*(1-w/K[2])-w*(c(2,0,u)+c(2,1,v))])
    return f
def jac(f,y,beta,eps=1e-6):
    J=np.zeros((3,3))
    for j in range(3):
        e=np.zeros(3);e[j]=eps; J[:,j]=(f(y+e,beta)-f(y-e,beta))/(2*eps)
    return J
def maxre(f,beta,guess):
    y=fsolve(f,guess,args=(beta,),xtol=1e-13); return max(np.linalg.eigvals(jac(f,y,beta)).real),y
# Part C
s,wk=3.15,0.05
a=[[0,wk,s],[s,0,wk],[wk,s,0]]; h=[[1]*3]*3
fC=make([1,1,1],[1,1,1],a,h,1.0)
print("C eq at beta=0", fsolve(fC,[0.29,0.28,0.29],args=(0,)), np.linalg.eigvals(jac(fC,fsolve(fC,[0.29,0.28,0.29],args=(0,)),0)))
bH=brentq(lambda b: maxre(fC,b,[0.29,0.28,0.29])[0],0.1,0.6); print("C beta_H",bH)
# Part D
a=[[0,0.95151628,2.02935451],[2.22193465,0,2.34064535],[0.45706170,2.76063376,0]]
h=[[0,0.61402428,0.14166076],[1.40938855,0,0.95875960],[2.08298487,2.03021611,0]]
fD=make([1,1,1],[0.5,1.0,0.8],a,h,1.0)
g=[0.02,0.32,0.36]
for b in [1.5,2.0,2.3,2.4]:
    m,y=maxre(fD,b,g); print("D",b,y,np.linalg.eigvals(jac(fD,y,b)))
bH=brentq(lambda b: maxre(fD,b,g)[0],2.0,2.9); print("D beta_H",bH)
# Turing check at beta=1.5
D=np.diag([0.287562209,0.000773583141,0.0000193559578]); m,y=maxre(fD,1.5,g); J=jac(fD,y,1.5)
qs=np.linspace(0,5,2001); sig=[max(np.linalg.eigvals(J-q*q*D).real) for q in qs]
i=int(np.argmax(sig)); band=qs[np.array(sig)>0]; print("Turing q*",qs[i],"sigma*",sig[i],"band",band.min(),band.max())
