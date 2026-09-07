"""Piskovsky (2025, Appl Math Lett 159:109269; arXiv:2405.14682) Lemma 4/5 test for N=3:
   Turing-Hopf  <=> exists y=k^2>0 with c1(y)>0 and c2(y)c1(y)-c0(y)>0
   Turing       <=> no Turing-Hopf and exists y>0 with c0(y)=det J(y) > 0
plus Satnoianu-Menzinger-Maini (2000) s-stability: all a_ii<0, all 2x2 minors>0, detJ<0 => no Turing for any D."""
import numpy as np, json, os
here=os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(here,'mechanism_test.py')).read().split('hits=[]')[0])
def coeffs(J,d,y):
    Jy=J-y*np.diag(d)
    c2=np.trace(Jy); c0=np.linalg.det(Jy)
    c1=sum(np.linalg.det(Jy[np.ix_([i,j],[i,j])]) for i,j in [(0,1),(0,2),(1,2)])
    return c2,c1,c0
def classify(J,d,ymax=100,n=20001):
    ys=np.linspace(1e-6,ymax,n); th=False; tu=False
    for y in ys:
        c2,c1,c0=coeffs(J,d,y)
        if c1>0 and c2*c1-c0>0: th=True
        if c0>0: tu=True
    minors=[np.linalg.det(J[np.ix_([i,j],[i,j])]) for i,j in [(0,1),(0,2),(1,2)]]
    sstable=all(np.diag(J)<0) and all(m>0 for m in minors) and np.linalg.det(J)<0
    return dict(turing_hopf=th, turing=(tu and not th), det_positive_somewhere=tu, s_stable=sstable,
                minors=np.round(minors,5).tolist(), diag=np.round(np.diag(J),5).tolist())
# Part D, beta=1.5, report diffusivities
a=[[0,0.95151628,2.02935451],[2.22193465,0,2.34064535],[0.45706170,2.76063376,0]]
h=[[0,0.61402428,0.14166076],[1.40938855,0,0.95875960],[2.08298487,2.03021611,0]]
fD=make([1,1,1],[0.5,1.0,0.8],a,h,1.0); y=fsolve(fD,[0.02,0.32,0.36],args=(1.5,),xtol=1e-13); J=jac(fD,y,1.5)
print("Part D beta=1.5, report D:", classify(J,[0.287562209,0.000773583141,0.0000193559578]))
print("Part D, v fast instead:  ", classify(J,[0.0000193559578,0.287562209,0.000773583141]))
# Part A beta=4 coexistence node
a=[[0,0.89,0.89],[0.55,0,0.5],[0.55,0.5,0]]; h=[[0,1,1],[0.5,0,1],[0.5,1,0]]
fA=make([1,0.8,0.8],[1,2,2],a,h,1.0); yA=fsolve(fA,[2.85,0.187,0.187],args=(4.0,),xtol=1e-13); JA=jac(fA,yA,4.0)
print("Part A beta=4, 14500-fold:", classify(JA,[0.29,0.0008,0.00002]))
# constructed example (trial 3868 from mechanism_test2)
hit=[x for x in json.load(open(os.path.join(here,'mechanism_hits.json'))) if x['trial']==3868][0]
f=make([1,1,1],hit['K'],hit['a'],hit['h'],1.0); y=fsolve(f,[0.7,0.25,0.1],args=(hit['beta'],),xtol=1e-13); J=jac(f,y,hit['beta'])
print("constructed 3868 eq",np.round(y,4),"u/K1",round(y[0]/hit['K'][0],2))
print("  engineer fast:", classify(J,[1.0,1e-3,1e-3]))
print("  v fast:       ", classify(J,[1e-3,1.0,1e-3]))
