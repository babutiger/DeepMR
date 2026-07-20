import time, cvxpy as cp, numpy as np
import deepzono_demo_5x50 as D
NNET='../../models/mnist_new_5x50/mnist_net_new_5x50.nnet'
def P(i): return '../../mnist_properties/mnist_properties_5x50/mnist_property_%d.txt'%i

print("### 1) CROSS-CHECK: DeepPoly-DeepMR (paper method) @ delta=0.0065, property 0")
net=D.network(); net.load_nnet(NNET); net.property_flag=False
t0=time.time(); vp=net.verify_lp_split(PROPERTY=P(0),DELTA=0.0065,MAX_ITER=5,SPLIT_NUM=0,WORKERS=8,TRIM=False,SOLVER=cp.GUROBI,MODE=1)
print("   DeepPoly-DeepMR @0.0065:", "VERIFIED" if vp else "UNVERIFIED", "(%.1fs)"%(time.time()-t0))

print("\n### 2) SOUNDNESS FALSIFICATION: at a delta with a real counterexample, DeepZono-DeepMR must NOT verify")
for d in [0.05, 0.08, 0.12]:
    net=D.network(); net.load_nnet(NNET); net.property_flag=False; net.load_robustness(P(0),d); net.deepzono()
    L0=net.layers[0]; los=np.array([n.concrete_lower for n in L0.neurons]); his=np.array([n.concrete_upper for n in L0.neurons])
    rng=np.random.RandomState(0); found=False; worstmax=-1e9
    cand=[los,his]+[los+(his-los)*rng.rand(len(los)) for _ in range(30000)]
    for x in cand:
        v=net.zono_forward_concrete(x); worstmax=max(worstmax,float(v.max()))
        if v.max()>0: found=True; break
    print("   delta=%.3f: random-attack counterexample found=%s (best halfspace=%.3f)"%(d,found,worstmax))
    if found:
        net2=D.network(); net2.load_nnet(NNET); net2.property_flag=False
        vz=net2.verify_zono(P(0),d,MAX_ITER=3,verbose=False)
        print("      -> DeepZono-DeepMR verified=%s  (MUST be False = sound)"%vz)
        break

print("\n### 3) ROBUSTNESS across properties: DeepZono-DeepMR @ delta=0.0065")
for i in [0,1,2,3]:
    net=D.network(); net.load_nnet(NNET); net.property_flag=False
    t0=time.time(); vz=net.verify_zono(P(i),0.0065,MAX_ITER=6,verbose=False)
    print("   property %d: %s (%.1fs)"%(i,"VERIFIED" if vz else "unverified",time.time()-t0))
