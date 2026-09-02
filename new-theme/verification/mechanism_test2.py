import numpy as np
from scipy.optimize import fsolve
exec(open(__import__('os').path.join(__import__('os').path.dirname(__file__),'mechanism_test.py')).read().split('hits=[]')[0])
rng=np.random.default_rng(1); hits=[]
for trial in range(4000):
    a=rng.uniform(0.2,3.0,(3,3)); np.fill_diagonal(a,0); h=rng.uniform(0.1,2.5,(3,3)); np.fill_diagonal(h,0)
    K=[rng.uniform(0.5,2),rng.uniform(0.5,2),rng.uniform(0.5,2)]; beta=rng.uniform(0.5,4)
    f=make([1,1,1],K,a,h,1.0)
    for g in [[0.5,0.5,0.5],[1,0.3,0.3],[0.3,0.8,0.2]]:
        y,info,ier,msg=fsolve(f,g,args=(beta,),full_output=True,xtol=1e-12)
        if ier!=1 or (y<0.02).any() or np.abs(f(y,beta)).max()>1e-9: continue
        J=jac(f,y,beta); ev=np.linalg.eigvals(J)
        if ev.real.max()>=0: continue
        m=minor(J,1,2)
        if m<0 and y[0]>0.3*K[0]:
            su,qu=turing(J,[1.0,1e-3,1e-3]); sv,_=turing(J,[1e-3,1.0,1e-3]); sw,_=turing(J,[1e-3,1e-3,1.0])
            hits.append(dict(trial=trial,y=np.round(y,3),u_over_K1=round(y[0]/K[0],2),vw_minor=round(m,4),
                 maxRe=np.round(ev.real.max(),4),complex=bool(np.iscomplex(ev).any()),
                 sig_u_fast=round(su,4),q=round(qu,2),sig_v_fast=round(sv,4),sig_w_fast=round(sw,4),
                 a=np.round(a,3).tolist(),h=np.round(h,3).tolist(),K=np.round(K,3).tolist(),beta=round(beta,3)))
            break
print("hits",len(hits))
pos=[x for x in hits if x['sig_u_fast']>0]; print("with Turing (u fast):",len(pos), " with Turing v-fast:",sum(x['sig_v_fast']>0 for x in hits)," w-fast:",sum(x['sig_w_fast']>0 for x in hits))
print("Turing hits where J has NO complex pair:",sum(not x['complex'] for x in pos))
for x in sorted(pos,key=lambda x:-x['u_over_K1'])[:5]: print(x)
import json; json.dump(hits,open('mechanism_hits.json','w'),default=str)
