#!/usr/bin/env python3
"""
Convert the DeepMR text properties into the VNN-LIB format used by VNN-COMP.

Each DeepMR property file lists the nominal input (one value per line) followed
by the unsafe half-spaces (one per line, ``w_0 ... w_{m-1} b`` meaning the output
``w . Y + b >= 0`` is a counterexample). This script writes, for every property,
an ``l_infinity`` local-robustness query:

  * input constraints  X_i in [x_i - eps, x_i + eps]  (clipped to [0,1] for
    normalized image inputs; left unclipped for the ACAS Xu control inputs);
  * an output constraint that is the disjunction of the unsafe half-spaces,
    so a satisfying assignment is exactly a counterexample to robustness.

Input dimension and number of outputs are detected automatically: the leading
single-token lines are the input, the remaining multi-token lines are the
half-spaces.
"""
import os


def _fmt(v):
    # compact float without trailing zeros / scientific noise
    return ("%.10g" % v)


def _halfspace_expr(weights, bias):
    """SMT-LIB '>= 0' assertion body for  sum_j w_j Y_j + bias >= 0.
    The frequent case (Y_a >= Y_b) is written in the short community form."""
    nz = [(j, w) for j, w in enumerate(weights) if w != 0.0]
    if bias == 0.0 and len(nz) == 2:
        (a, wa), (b, wb) = nz
        if wa == 1.0 and wb == -1.0:
            return "(>= Y_%d Y_%d)" % (a, b)
        if wa == -1.0 and wb == 1.0:
            return "(>= Y_%d Y_%d)" % (b, a)
    terms = ["(* %s Y_%d)" % (_fmt(w), j) for j, w in nz]
    if bias != 0.0:
        terms.append(_fmt(bias))
    inner = terms[0] if len(terms) == 1 else "(+ %s)" % " ".join(terms)
    return "(>= %s 0)" % inner


def convert_file(txt_path, out_path, eps, clip):
    lines = [ln.strip() for ln in open(txt_path) if ln.strip() != ""]
    inp, halfspaces = [], []
    for ln in lines:
        toks = ln.split()
        if len(toks) == 1 and not halfspaces:
            inp.append(float(toks[0]))
        else:
            halfspaces.append([float(t) for t in toks])
    n_in = len(inp)
    n_out = len(halfspaces[0]) - 1
    with open(out_path, "w") as f:
        f.write("; %s  (eps = %s)\n\n" % (os.path.basename(txt_path), eps))
        for i in range(n_in):
            f.write("(declare-const X_%d Real)\n" % i)
        f.write("\n")
        for i in range(n_out):
            f.write("(declare-const Y_%d Real)\n" % i)
        f.write("\n; Input constraints:\n")
        for i, x in enumerate(inp):
            lb, ub = x - eps, x + eps
            if clip:
                lb, ub = max(0.0, lb), min(1.0, ub)
            f.write("(assert (<= X_%d %s))\n" % (i, _fmt(ub)))
            f.write("(assert (>= X_%d %s))\n" % (i, _fmt(lb)))
        f.write("\n; Output constraints (a counterexample satisfies one of the unsafe half-spaces):\n")
        f.write("(assert (or\n")
        for hs in halfspaces:
            f.write("    (and %s)\n" % _halfspace_expr(hs[:-1], hs[-1]))
        f.write("))\n")
    return n_in, n_out, len(halfspaces)


def convert_dir(in_dir, out_dir, eps, clip):
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    info = None
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".txt"):
            continue
        info = convert_file(os.path.join(in_dir, name),
                            os.path.join(out_dir, name[:-4] + ".vnnlib"),
                            eps, clip)
        n += 1
    return n, info


# (input folder, output folder, eps/radius, clip-to-[0,1])
JOBS = [
    ("mnist_properties/mnist_properties_5x50",  "mnist_5x50",  0.018, True),
    ("mnist_properties/mnist_properties_5x80",  "mnist_5x80",  0.019, True),
    ("mnist_properties/mnist_properties_6x100", "mnist_6x100", 0.019, True),
    ("mnist_properties/mnist_properties_9x100", "mnist_9x100", 0.018, True),
    ("mnist_properties/mnist_properties_10x80", "mnist_10x80", 0.015, True),
    ("cifar_properties/cifar_properties_5x50",  "cifar_5x50",  0.0030, True),
    ("cifar_properties/cifar_properties_6x80",  "cifar_6x80",  0.0038, True),
    ("cifar_properties/cifar_properties_10x100","cifar_10x100",0.0027, True),
    ("vnncomp_eran_properties/vnncomp_eran_mnist_properties_6x100", "eran_mnist_5x100", 0.019, True),
    ("vnncomp_eran_properties/vnncomp_eran_mnist_properties_9x100", "eran_mnist_8x100", 0.018, True),
    ("vnncomp_eran_properties/vnncomp_eran_mnist_properties_6x200", "eran_mnist_5x200", 0.013, True),
    ("vnncomp_eran_properties/vnncomp_eran_mnist_properties_9x200", "eran_mnist_8x200", 0.012, True),
    ("acas_properties/acas_xu_p5_net_1_1", "acasxu_phi5_net_1_1", 2.0,   False),
    ("acas_properties/acas_xu_p6_net_1_1", "acasxu_phi6_net_1_1", 120.0, False),
    ("acas_properties/acas_xu_p7_net_1_9", "acasxu_phi7_net_1_9", 6.0,   False),
]

if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(root)
    for in_rel, out_name, eps, clip in JOBS:
        in_dir = os.path.join(repo, in_rel)
        if not os.path.isdir(in_dir):
            print("SKIP (missing): %s" % in_rel)
            continue
        n, info = convert_dir(in_dir, os.path.join(root, out_name), eps, clip)
        print("%-22s eps=%-7s clip=%-5s -> %4d files  (in=%d out=%d hs=%d)"
              % (out_name, eps, clip, n, info[0], info[1], info[2]))
