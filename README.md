# Netzwerk

A collection of state-of-the-art contraction ordering algorithms

## Description

Netzwork hosts optimal and near-optimal algorithms for tensor network contraction ordering:

* $\texttt{TensorIKKBZ}$: Optimal linear contraction orders for tree tensor networks

## Setup

### Setup `opt_einsum`

```
cd third-party/opt_einsum
pip3 install -e .
cd ../..
```

### Setup `cotengra`

```
cd third-party/cotengra
pip3 install -e .
cd ../..
```

### Setup our framework

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

### Setup `kahypar`

Follow the instructions provided [here](https://kahypar.org/). In particular, consider installing manually, as working directly with the `pip` package, while easy to install, may throw errors, e.g., `concurrent.futures.process.BrokenProcessPool`.

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
