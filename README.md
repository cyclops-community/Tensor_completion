## Distributed-Memory Tensor Completion Algorithms

This repository contains tensor completion algorithms (ALS with explicit and implicit CG, SGD, and CCD++) implemented using the Python interface to Cyclops.

## Dependencies

Usage of these codes requires installation of the [Cyclops library](https://github.com/cyclops-community/ctf) and its python extension..

## Execution

To benchmark and test the codes run `combined_test.py`, which maybe used as follows.
```
python ./combined_test.py --help
usage: combined_test.py [-h] [--I int] [--J int] [--K int] [--R int]
                        [--num-iter-ALS-implicit int]
                        [--num-iter-ALS-explicit int] [--num-iter-CCD int]
                        [--num-iter-SGD int] [--time-limit float]
                        [--obj-freq-CCD int] [--obj-freq-SGD int]
                        [--err-thresh float] [--sp-fraction float]
                        [--use-sparse-rep int] [--block-size-ALS-implicit int]
                        [--block-size-ALS-explicit int]
                        [--regularization-ALS float]
                        [--regularization-CCD float]
                        [--regularization-SGD float] [--learning-rate float]
                        [--sample-frac-SGD float] [--function-tensor int]
                        [--use-CCD-TTTP int] [--tensor-file str]

optional arguments:
  -h, --help            show this help message and exit
  --I int               Input tensor size in first dimension (default: 64)
  --J int               Input tensor size in second dimension (default: 64)
  --K int               Input tensor size in third dimension (default: 64)
  --R int               Input CP decomposition rank (default: 10)
  --num-iter-ALS-implicit int
                        Number of iterations (sweeps) to run ALS with implicit
                        CG (default: 10)
  --num-iter-ALS-explicit int
                        Number of iterations (sweeps) to run ALS with explicit
                        CG (default: 10)
  --num-iter-CCD int    Number of iterations (updates to each column of each
                        matrix) for which to run CCD (default: 10)
  --num-iter-SGD int    Number of iteration, each iteration computes
                        subgradients from --sample-frac-SGD of the total
                        number of nonzeros in tensor (default: 10)
  --time-limit float    Number of seconds after which to terminate tests for
                        either ALS, SGD, or CCD if number of iterations is not
                        exceeded (default: 30)
  --obj-freq-CCD int    Number of iterations after which to calculate
                        objective (time for objective calculation not included
                        in time limit) for CCD (default: 1)
  --obj-freq-SGD int    Number of iterations after which to calculate
                        objective (time for objective calculation not included
                        in time limit) for SGD (default: 1)
  --err-thresh float    Residual norm threshold at which to halt if number of
                        iterations does not expire (default 1.E-5)
  --sp-fraction float   sparsity (default: .1)
  --use-sparse-rep int  whether to store tensor as sparse (default: 1, i.e.
                        True)
  --block-size-ALS-implicit int
                        block-size for implicit ALS (default: 0, meaning to
                        use a single block)
  --block-size-ALS-explicit int
                        block-size for explicit ALS (default: 0, meaning to
                        use a single block)
  --regularization-ALS float
                        regularization for ALS (default: 0.00001)
  --regularization-CCD float
                        regularization for CCD (default: 0.00001)
  --regularization-SGD float
                        regularization for SGD (default: 0.00001)
  --learning-rate float
                        learning rate for SGD (default: 0.01)
  --sample-frac-SGD float
                        sample size as fraction of total number of nonzeros
                        for SGD (default: 0.01)
  --function-tensor int
                        whether to use function tensor as test problem
                        (default: 0, i.e. False, use explicit low CP-rank
                        sampled tensor)
  --use-CCD-TTTP int    whether to use TTTP for CCD contractions (default: 1,
                        i.e. Yes)
  --tensor-file str     Filename from which to read tensor (default: None, use
                        model problem)
```
To execute with MPI (for example with 4 processes), use `mpirun -np 4 python ./combined_test.py ...`.

Some scripts to obtain and preprocess various test datasets are provided in the `data/` directory.
