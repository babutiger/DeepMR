# Conversion scripts

Format-conversion utilities used to run the baseline tools and to export the
properties, so that the same benchmarks can be verified across DeepMR and the
community tools. Paths in the example calls at the bottom of each script are
relative placeholders — adjust them to your layout (for MNIST the input size is
784, for CIFAR-10 it is 3072).

## Property conversion (VNN-LIB)

| Script | Purpose |
| --- | --- |
| `vnnlib_batch_sym2VNNLIB_files.py` | Batch-convert a folder of DeepMR text properties to VNN-LIB (`l_infinity` robustness). |
| `vnnlib_batch_VNNLIB_files.py` | Batch conversion variant. |
| `vnnlib_single_VNNLIB_file.py` | Convert a single property to VNN-LIB. |
| `vnnlib2csv_rootdir.py` | Collect the generated VNN-LIB files into a `instances.csv` list (paths from the root). |
| `vnnlib2csv_relativedir.py` | Same, with relative paths. |

A cleaner, self-contained round-trip converter is also provided under
[`../vnnlib_properties/`](../vnnlib_properties) (`convert_to_vnnlib.py` and
`vnnlib_to_csv.py`), together with the generated VNN-LIB files.

## Baseline-tool formats

| Script | Purpose |
| --- | --- |
| `test_ERAN_all_csv.py`, `test_ERAN_csv.py` | Export properties to the CSV format read by ERAN / PRIMA. |
| `test_marabou_txtfile2.py`, `test_marabou_txtfile3_flod.py`, `test_marabou_dayuhao.py` | Rewrite properties into the Marabou query format. |
| `newpth_mnist_10x80.py` | Re-encode a trained model (layer-wise) for the baseline pipelines. |
