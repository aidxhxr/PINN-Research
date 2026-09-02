"""Constructive test of the corrected Turing mechanism:
Part-A symmetric structure, but v-w competition strengthened so that v,w alone are a
founder-control (saddle) pair. Look for a stable interior equilibrium with vw-minor<0,
then scan diffusivities: engineer fast should pattern; competitor fast should not."""
import numpy as np, itertools
from scipy.optimize import fsolve
def make(r,K,a,h,gam):
    def f(y,beta):
        u,v,w=y; Ku=K[0]+beta*u/(1+gam*u)
        c=lambda i,j,N: a[i][j]*N/(1+h[i][j]*N)
        return np.array([r[0]*u*(1-u/Ku)-u*(c(0,1,v)+c(0,2,w)),
                         r[1]*v*(1-v/K[1])-v*(c(1,0,u)+c(1,2,w)),
                         r[2]*w*(1-w/K[2])-w*(c(2,0,u)+c(2,1,v))])
    return f
def jac(f,y,b,eps=1e-6):
    J=np.zeros((3,3))
    for j in range(3):
        e=np.zeros(3);e[j]=eps; J[:,j]=(f(y+e,b)-f(y-e,b))/(2*eps)
    return J
def minor(J,i,j): return np.linalg.det(J[np.ix_([i,j],[i,j])])
def turing(J,D,qmax=20,n=4001):
    qs=np.linspace(0,qmax,n); s=np.array([max(np.linalg.eigvals(J-q*q*np.diag(D)).real) for q in qs])
    return s.max(), qs[s.argmax()]
hits=[]
for a23 in [0.5,0.8,1.0,1.2,1.5,2.0]:
  for a12 in [0.6,0.89,1.2]:
    for a21 in [0.3,0.55,0.8]:
      a=[[0,a12,a12],[a21,0,a23],[a21,a23,0]]; h=[[0,1,1],[0.5,0,1],[0.5,1,0]]
      f=make([1,0.8,0.8],[1,2,2],a,h,1.0)
      for beta in [1,2,3,4,5]:
        for g in [[2,0.3,0.3],[1,0.5,0.5],[0.5,1,1],[3,0.1,0.1]]:
          y,info,ier,msg=fsolve(f,g,args=(beta,),full_output=True,xtol=1e-12)
          if ier!=1 or (y<1e-3).any(): continue
          J=jac(f,y,beta); ev=np.linalg.eigvals(J)
          if ev.real.max()>=0: continue
          m=minor(J,1,2)
          if m<0:
            su,_=turing(J,[1.0,1e-3,1e-3]); sv,_=turing(J,[1e-3,1.0,1e-3])
            hits.append((a12,a21,a23,beta,tuple(np.round(y,3)),round(m,5),round(ev.real.max(),4),round(su,5),round(sv,5)))
            break
seen=set()
print("a12 a21 a23 beta | eq | vw-minor | maxRe(J) | sigma* engineer-fast | sigma* v-fast")
for hset in hits:
    if hset[:4] in seen: continue
    seen.add(hset[:4]); print(hset)
print("stable interior eq with vw-minor<0 found:",len(seen))
