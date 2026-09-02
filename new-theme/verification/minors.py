import numpy as np
exec(open('verify2.py').read().split('# Turing check')[0].replace('print(','(lambda *a,**k:None)('))
def minors(J):
    return {"uv":np.linalg.det(J[np.ix_([0,1],[0,1])]),"uw":np.linalg.det(J[np.ix_([0,2],[0,2])]),"vw":np.linalg.det(J[np.ix_([1,2],[1,2])])}
# Part D beta=1.5
m,y=maxre(fD,1.5,g); J=jac(fD,y,1.5)
print("PartD J diag",np.diag(J).round(5)); print("PartD minors",{k:round(v,6) for k,v in minors(J).items()}, "detJ",np.linalg.det(J))
# Part A beta=4 coexistence node
a=[[0,0.89,0.89],[0.55,0,0.5],[0.55,0.5,0]]; h=[[0,1,1],[0.5,0,1],[0.5,1,0]]
fA=make([1,0.8,0.8],[1,2,2],a,h,1.0); m,yA=maxre(fA,4.0,[2.85,0.187,0.187]); JA=jac(fA,yA,4.0)
print("PartA J diag",np.diag(JA).round(5)); print("PartA minors",{k:round(v,6) for k,v in minors(JA).items()}, "detJ",np.linalg.det(JA))
# Test: does a Turing band exist in Part D when we fix D_w tiny, D_v tiny but D_u big? which species must be fast?
D0=[0.287562209,0.000773583141,0.0000193559578]
for name,D in [("as report",D0),("u slow, v fast",[D0[2],D0[0],D0[1]]),("w fast",[D0[1],D0[2],D0[0]]),("u fast only, v=w tiny equal",[0.29,2e-5,2e-5])]:
    Dm=np.diag(D); qs=np.linspace(0,8,4001); sig=np.array([max(np.linalg.eigvals(J-q*q*Dm).real) for q in qs])
    print(name, "max sigma", sig.max().round(6), "at q", qs[sig.argmax()].round(3))
