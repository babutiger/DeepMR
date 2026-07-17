import sys, os, statistics as st
sys.path.insert(0, '.')
import deeppoly_mnist_new_6x100 as DP   # generic network class

# whole-network max M_min: M_min = max_i(-lower_bound(margin_i)), computed from DeepPoly output bounds (single pass)
NPROPS = int(os.environ.get("NPROPS", "100"))
# (name, model, propdir, prefix, eps, n)
NETS = [
    ("M5a", "../../models/mnist_new_5x50/mnist_net_new_5x50.nnet",   "../../mnist_properties/mnist_properties_5x50",  "mnist_property", 0.018, 100),
    ("M5b", "../../models/mnist_new_5x80/mnist_net_new_5x80.nnet",   "../../mnist_properties/mnist_properties_5x80",  "mnist_property", 0.019, 100),
    ("M6",  "../../models/mnist_new_6x100/mnist_net_new_6x100.nnet", "../../mnist_properties/mnist_properties_6x100", "mnist_property", 0.019, 100),
    ("M9",  "../../models/mnist_new_9x100/mnist_net_new_9x100.nnet", "../../mnist_properties/mnist_properties_9x100", "mnist_property", 0.018, 100),
    ("M10", "../../models/mnist_new_10x80/mnist_net_new_10x80.nnet", "../../mnist_properties/mnist_properties_10x80", "mnist_property", 0.015, 100),
    ("C5",  "../../models/cifar_new_5x50/cifar_net_new_5x50.nnet",   "../../cifar_properties/cifar_properties_5x50",  "cifar_property", 0.0030, 100),
    ("C6",  "../../models/cifar_new_6x80/cifar_net_new_6x80.nnet",   "../../cifar_properties/cifar_properties_6x80",  "cifar_property", 0.0038, 100),
    ("C10", "../../models/cifar_new_10x100/cifar_net_new_10x100.nnet","../../cifar_properties/cifar_properties_10x100","cifar_property", 0.0027, 100),
]
print("net,eps,n,max_Mmin,mean_Mmin,n_over_10000")
grand = 0.0
for name, model, pdir, pref, eps, npr in NETS:
    npr = min(npr, NPROPS)
    mins = []
    for i in range(npr):
        pf = f"{pdir}/{pref}_{i}.txt"
        if not os.path.exists(pf):
            continue
        try:
            net = DP.network(); net.load_nnet(model)
            net.load_robustness(pf, eps, TRIM=True)
            net.deeppoly()
            outs = net.layers[-1].neurons
            adv = [n for n in outs if n.concrete_upper is not None and n.concrete_upper >= 0]
            mins.append(max([-n.concrete_lower for n in adv], default=0.0))
        except Exception as e:
            print(f"  {name} prop_{i} ERR {type(e).__name__} {str(e)[:60]}", flush=True)
    if mins:
        mx = max(mins); grand = max(grand, mx)
        over = sum(1 for m in mins if m > 10000)
        print(f"{name},{eps},{len(mins)},{mx:.1f},{st.mean(mins):.1f},{over}", flush=True)
print(f"\n[RESULT] GRAND max M_min over all nets = {grand:.1f}")
print(f"[RESULT] M=10000 covers all instances: {grand <= 10000}")
print("[MMINALL] DONE", flush=True)
