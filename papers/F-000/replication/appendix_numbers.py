"""
Reproducible numerics for F-000 appendices / SPTLS section.
Run: python3 appendix_numbers.py
Deterministic (seed 7 + 2-decimal rounding => reproducible from printed series).
"""
import numpy as np

# ---------- 1. Toy series for the worked OLS (no leakage) ----------
rng=np.random.default_rng(7); T=22
z=np.zeros(T); z[0]=2.0
for t in range(1,T):
    z[t]=0.6+0.62*z[t-1]+0.10*np.sin(1.3*t)+0.08*rng.standard_normal()
z=np.round(z,2)

q=np.vstack([z[2:], z[2:]-z[1:-1], z[2:]-2*z[1:-1]+z[:-2]]).T   # q(t)
qa,qb,qc=q[:-2],q[1:-1],q[2:]                                   # q(t-1),q(t),q(t+1)
# LAGGED bivector B^-(t)=q(t-1)∧q(t)  (measurable at t; NO leakage)
Bm12=qa[:,0]*qb[:,1]-qb[:,0]*qa[:,1]
Bm13=qa[:,0]*qb[:,2]-qb[:,0]*qa[:,2]
y=qc[:,1]                                                       # target Δz_{t+1}
n=len(y)
X =np.column_stack([np.ones(n), qb[:,0], qb[:,1], Bm12, Bm13])  # SPTLS (with grade-2)
Xr=X[:,:3]                                                      # OLS-AR(1) projection
th ,*_=np.linalg.lstsq(X ,y,rcond=None)
thr,*_=np.linalg.lstsq(Xr,y,rcond=None)
rss =np.sum((y-X @th )**2); rssr=np.sum((y-Xr@thr)**2); tss=np.sum((y-y.mean())**2)
print("z_t:", " ".join(f"{v:.2f}" for v in z))
print(f"[OLS no-leak] n={n} K=5 cond(XtX)={np.linalg.cond(X.T@X):.2e}")
print(f"  SPTLS  theta[const,z,Δz,B-12,B-13]={np.round(th,3)}  R2={1-rss/tss:.3f}")
print(f"  OLS-AR theta[const,z,Δz]          ={np.round(thr,3)}  R2={1-rssr/tss:.3f}")
print(f"  grade-2 RSS reduction: {100*(rssr-rss)/rssr:.1f}%")

# ---------- 2. Non-abelian matrix/rotor fit (Lorenz) ----------
def lorenz(N=4000,h=0.01,s=10,r=28,b=8/3):
    x,y_,zz=1.,1.,1.; xs=[]
    for _ in range(N+200):
        x,y_,zz=x+h*s*(y_-x),y_+h*(x*(r-zz)-y_),zz+h*(x*y_-b*zz); xs.append(x)
    return np.array(xs[200:])
xs=lorenz()
QL=np.vstack([xs[2:],xs[2:]-xs[1:-1],xs[2:]-2*xs[1:-1]+xs[:-2]]).T
a0,a1=QL[:-1],QL[1:]
Mt,*_=np.linalg.lstsq(a0,a1,rcond=None); M=Mt.T          # q(t+1)=M q(t)
U,S,Vt=np.linalg.svd(M); R=U@Vt
if np.linalg.det(R)<0: U=U.copy(); U[:,-1]*=-1; R=U@Vt
ang=np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
rfull=np.sum((a1-a0@Mt)**2)
Bs=[np.zeros((3,3)) for _ in range(6)]
for E,(i,j) in zip(Bs,[(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]): E[i,j]=1;E[j,i]=1
A=np.vstack([(a0@E).reshape(-1) for E in Bs]).T
c,*_=np.linalg.lstsq(A,a1.reshape(-1),rcond=None)
Ssym=sum(ck*E for ck,E in zip(c,Bs)); rsym=np.sum((a1-a0@Ssym.T)**2)
print(f"[matrix/rotor Lorenz] polar rotation={ang:.1f} deg")
print(f"  resid full M={rfull:.3e}  resid symmetric(abelian)={rsym:.3e}  non-abelian drop={100*(rsym-rfull)/rsym:.2f}%")
print(f"  ||M-M^T||/||M||={np.linalg.norm(M-M.T)/np.linalg.norm(M):.3f}")

# ---------- 3. Weyl [D,t]=1 collinearity ----------
tt=np.arange(1,T).astype(float); zk=z[1:]; zkm=z[:-1]; dz=zk-zkm
c1=tt*zk-(tt-1)*zkm; c2=tt*dz; c3=zkm
Xdep=np.vstack([c1,c2,c3]).T; Xok=np.vstack([c2,c3]).T
print(f"[Weyl] max|Δ(tz)-(tΔz+z)|={np.max(np.abs(c1-(c2+c3))):.1e}  rank{{Δ(tz),tΔz,z}}={np.linalg.matrix_rank(Xdep)}/3")
print(f"  cond(dependent)={np.linalg.cond(Xdep.T@Xdep):.1e}  cond(normal-ordered {{tΔz,z}})={np.linalg.cond(Xok.T@Xok):.1e}")
