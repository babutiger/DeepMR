import time, cvxpy as cp, numpy as np
import deepzono_demo_5x50 as D
NNET='../../models/mnist_new_5x50/mnist_net_new_5x50.nnet'
def P(i): return '../../mnist_properties/mnist_properties_5x50/mnist_property_%d.txt'%i

def fwd_jac(net, x):
    vals=np.array(x,float); J=np.eye(len(x))
    for k in range(1,len(net.layers)):
        cur=net.layers[k]
        if cur.layer_type==1:  # AFFINE
            W=np.stack([n.weight for n in cur.neurons]); b=np.array([n.bias for n in cur.neurons])
            vals=W@vals+b; J=W@J
        elif cur.layer_type==2: # RELU
            m=(vals>0).astype(float); vals=vals*m; J=m[:,None]*J
    return vals,J  # verify halfspaces + jacobian

def pgd(net,i,d,steps=60):
    net.property_flag=False; net.load_robustness(P(i),d)
    L0=net.layers[0]; lo=np.array([n.concrete_lower for n in L0.neurons]); hi=np.array([n.concrete_upper for n in L0.neurons])
    x=0.5*(lo+hi); best=-1e9
    for _ in range(steps):
        v,J=fwd_jac(net,x); best=max(best,float(v.max()))
        if v.max()>1e-6: return True,best
        wi=int(np.argmax(v)); g=J[wi]; x=np.clip(x+ (hi-lo)*0.5*np.sign(g)*0.3, lo,hi)
    v,_=fwd_jac(net,x); return (v.max()>1e-6), max(best,float(v.max()))

print("### DeepPoly-DeepMR @0.0065 property 2 (expect VERIFIED => DeepZono 'unverified' there is precision, not unsound)")
net=D.network(); net.load_nnet(NNET); net.property_flag=False
vp2=net.verify_lp_split(PROPERTY=P(2),DELTA=0.0065,MAX_ITER=5,SPLIT_NUM=0,WORKERS=8,TRIM=False,SOLVER=cp.GUROBI,MODE=1)
print("   DeepPoly-DeepMR @0.0065 prop2:", "VERIFIED" if vp2 else "UNVERIFIED")

print("\n### SOUNDNESS: PGD find real counterexample, then DeepZono-DeepMR must say UNVERIFIED")
net=D.network(); net.load_nnet(NNET)
for d in [0.15,0.2,0.25,0.3]:
    found,best=pgd(net,0,d)
    print("   prop0 delta=%.2f: PGD counterexample=%s (best halfspace=%.3f)"%(d,found,best))
    if found:
        net2=D.network(); net2.load_nnet(NNET); net2.property_flag=False
        vz=net2.verify_zono(P(0),d,MAX_ITER=4,verbose=False)
        print("      -> DeepZono-DeepMR verified=%s  (MUST be False = sound)"%vz)
        break
