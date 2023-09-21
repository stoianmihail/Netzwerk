# <code style="color:#0065BD">Netzwerk</code>

A collection of state-of-the-art contraction ordering algorithms. Plug-in for [`opt_einsum`](https://github.com/dgasmith/opt_einsum) and [`cotengra`](https://github.com/jcmgray/cotengra).
- Optimal and near-optimal algorithms for several classes of tensor networks
- Based on the latest research in contraction ordering and database join ordering
- Careful C++ implementations

## Description

$\large \textcolor{#0065bd}{\texttt{Netzwerk}}$ hosts optimal and near-optimal algorithms for tensor network contraction ordering:

* $\large \texttt{TensorIKKBZ}$: Optimal linear contraction orders for tree tensor networks
* $\large \texttt{LinDP}$: Optimal general contraction trees _given_ the linear contraction orders of $\large \texttt{TensorIKKBZ}$
* Coming soon..

## Setup

Build the shared library, which requires [`CMake`](https://cmake.org).

```
cd src/netzwerk
mkdir -p build
cd build
cmake ..
make
```

The following initializes the package, assuming you are still in `build`:

```
cd ../../
versioneer install
pip3 install -e .
```

## Generation

Example: generate FTPS of up to 100 tensors.

```
python3 gen.py -t ftps -s 100
```

The files will be stored in `data/ftps`.

## Run

The script `bench.sh` runs the experiments. For instance, if you want to benchmark on the networks you have just generated:

```
./bench.sh opt_einsum ftps 100
```

The results will be stored in `results`. For the Sycamore circuit, simply run

```
./bench.sh cotengra sycamore
```

## Plot

Use the following notebooks to plot the results:
* `bench-plot.ipynb`: for networks generated via `gen.py`
* `circuit-plot.ipynb`: for quantum circuits, e.g., Sycamore. 
