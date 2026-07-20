import sys, os, statistics as st
sys.path.insert(0, '.')
import deeppoly_mnist_new_6x100 as DP   # 通用 network 类(utils.util import正确), load_nnet 自动解析架构

# 用 DeepPoly 扫 ERAN 6x200/9x200 每实例最大可验证半径(ε), 供选"不大不小"的 ε
# ε 单位 /1000; R=60 => 上限 0.06; TRIM=True 与 DeepMR 跑法一致
NPROPS = int(os.environ.get("SCAN_NPROPS", "40"))
NETS = {
    "6x200": ("../../models/mnist_new_6x200/mnist_6_200_nat_flat.nnet",
              "../../vnncomp_eran_properties/vnncomp_eran_mnist_properties_6x200"),
    "9x200": ("../../models/mnist_new_9x200/mnist_9_200_nat_flat.nnet",
              "../../vnncomp_eran_properties/vnncomp_eran_mnist_properties_9x200"),
}
for name, (model, pdir) in NETS.items():
    radii = []
    for i in range(NPROPS):
        net = DP.network(); net.load_nnet(model)
        r = net.find_max_disturbance(PROPERTY=f"{pdir}/mnist_property_{i}.txt", L=0, R=60, TRIM=True)
        radii.append(r)
        print(f"[{name}] prop_{i} deeppoly_max_radius={r}", flush=True)
    rs = sorted(radii); n = len(rs)
    print(f"[{name}] SUMMARY n={n} min={rs[0]:.3f} p25={rs[n//4]:.3f} "
          f"median={st.median(rs):.3f} p60={rs[int(n*0.6)]:.3f} p75={rs[3*n//4]:.3f} max={rs[-1]:.3f}", flush=True)
print("[SCAN] DONE", flush=True)
