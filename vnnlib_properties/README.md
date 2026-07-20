# VNN-LIB properties

The verification properties used in the paper, exported to the **VNN-LIB**
format used by [VNN-COMP](https://github.com/verivital/vnn-comp), so that the
benchmarks are interoperable with other community tools. Both conversion
directions are provided:

* `convert_to_vnnlib.py` — DeepMR text properties → VNN-LIB (regenerates this folder);
* `vnnlib_to_csv.py` — VNN-LIB → the internal CSV representation used by DeepMR.

Each property is an `l_infinity` local-robustness query: input constraints
`X_i in [x_i - eps, x_i + eps]` (clipped to `[0,1]` for the normalized image
inputs, left unclipped for the ACAS Xu control inputs), and an output constraint
that is the disjunction of the unsafe half-spaces, so a satisfying assignment is
exactly a counterexample to robustness.

| Folder | Network | Inputs | eps |
| --- | --- | --- | --- |
| `mnist_5x50`, `mnist_5x80`, `mnist_6x100`, `mnist_9x100`, `mnist_10x80` | MNIST FC-ReLU (M5a–M10) | 784 | 0.018, 0.019, 0.019, 0.018, 0.015 |
| `cifar_5x50`, `cifar_6x80`, `cifar_10x100` | CIFAR-10 FC-ReLU (C5, C6, C10) | 3072 | 0.0030, 0.0038, 0.0027 |
| `eran_mnist_5x100`, `eran_mnist_8x100`, `eran_mnist_5x200`, `eran_mnist_8x200` | VNN-COMP ERAN MNIST | 784 | 0.019, 0.018, 0.013, 0.012 |
| `acasxu_phi5_net_1_1`, `acasxu_phi6_net_1_1`, `acasxu_phi7_net_1_9` | ACAS Xu (phi5–phi7) | 5 | 2, 120, 6 |

To regenerate:

```bash
python3 convert_to_vnnlib.py
```
