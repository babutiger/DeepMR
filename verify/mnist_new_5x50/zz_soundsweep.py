import cvxpy as cp, numpy as np
import deepzono_demo_5x50 as D
NNET='../../models/mnist_new_5x50/mnist_net_new_5x50.nnet'
def P(i): return '../../mnist_properties/mnist_properties_5x50/mnist_property_%d.txt'%i
def fwd_jac(net,x):
    vals=np.array(x,float); J=np.eye(len(x))
    for k in range(1,len(net.layers)):
        c=net.layers[k]
        if c.layer_type==1:
            W=np.stack([n.weight for n in c.neurons]); b=np.array([n.bias for n in c.neurons]); vals=W@vals+b; J=W@J
        elif c.layer_type==2:
            m=(vals>0).astype(float); vals=vals*m; J=m[:,None]*J
    return vals,J
def pgd(net,i,d,steps=80):
    net.property_flag=False; net.load_robustness(P(i),d)
    L0=net.layers[0]; lo=np.array([n.concrete_lower for n in L0.neurons]); hi=np.array([n.concrete_upper for n in L0.neurons])
    x=0.5*(lo+hi); best=-1e9
    for _ in range(steps):
        v,J=fwd_jac(net,x); best=max(best,float(v.max()))
        if v.max()>1e-6: return True
        wi=int(np.argmax(v)); x=np.clip(x+(hi-lo)*0.5*np.sign(J[wi])*0.25,lo,hi)
    return best>1e-6
print("SOUNDNESS SWEEP @ delta=0.15 (properties 0-6): if a real counterexample exists, DeepZono-DeepMR must be UNVERIFIED")
bad=0
for i in range(7):
    net=D.network(); net.load_nnet(NNET)
    ce=pgd(net,i,0.15)
    net2=D.network(); net2.load_nnet(NNET); net2.property_flag=False
    vz=net2.verify_zono(P(i),0.15,MAX_ITER=2,verbose=False)
    ok = not (ce and vz)   # unsound iff counterexample exists but zono verified
    if not ok: bad+=1
    print("  prop %d: real_counterexample=%s  DeepZono-DeepMR_verified=%s  %s"%(i,ce,vz,"OK" if ok else "*** UNSOUND ***"))
print("UNSOUND cases: %d / 7  (0 = sound)"%bad)
