import sys, os
sys.path.insert(0, '.')
import deeppoly_mnist_new_6x100 as DP   # 通用类, utils.util import 正确

# 计算每实例 Big-M 的理论最小有效值 M_min = max_i(-下界(margin_i))
# 依据 Big-M 标准: M 必须 >= 约束表达式在可行域上的最大值(这里由 DeepPoly 输出边界给出保守上界)
MODEL = "../../models/mnist_new_6x100/mnist_net_new_6x100.nnet"   # M6 (论文自训)
PDIR  = "../../mnist_properties/mnist_properties_6x100"
EPS   = 0.019
VERIF = [11, 12, 49, 52, 55, 58, 79, 95]   # 靠精化验证成功的
FAIL  = [0, 2, 4]                           # 精化后仍失败的(有真实反例)
print("group,prop,M_min,max_abs_output_bound,n_adv_neurons,deeppoly_verified")
allmin = []
for group, plist in [("verified", VERIF), ("failing", FAIL)]:
    for p in plist:
        net = DP.network(); net.load_nnet(MODEL)
        net.load_robustness(f"{PDIR}/mnist_property_{p}.txt", EPS, TRIM=True)
        net.deeppoly()
        outs = net.layers[-1].neurons
        adv = [n for n in outs if n.concrete_upper is not None and n.concrete_upper >= 0]
        Mmin = max([-n.concrete_lower for n in adv], default=0.0)
        maxabs = max([max(abs(n.concrete_lower), abs(n.concrete_upper)) for n in outs], default=0.0)
        verified = all(n.concrete_upper <= 0 for n in outs)
        allmin.append(Mmin)
        print(f"{group},{p},{Mmin:.4f},{maxabs:.4f},{len(adv)},{verified}", flush=True)
print(f"\n[RESULT] max M_min over all instances = {max(allmin):.4f}")
print(f"[RESULT] 论文用的 M=10000 是否 >= 所有 M_min: {10000 >= max(allmin)}  (倍数={10000/max(allmin):.1f}x)")
