---
layout: default
title: PySPOD
tagline: A parallel SPOD package
---

PySPOD is a **parallel (distributed) library**, that implements the
**spectral proper orthogonal decomposition**, briefly **SPOD**, in
Python. It is based on the serial Matlab implementations of the classical [batch algorithm](https://www.mathworks.com/matlabcentral/fileexchange/65683-spectral-proper-orthogonal-decomposition-spod) and the [streaming version](https://www.mathworks.com/matlabcentral/fileexchange/69963-streaming-spectral-proper-orthogonal-decomposition).

The library uses [mpi4py](https://github.com/mpi4py/mpi4py) to deploy
parallel capabilities. This needs a version of MPI (Message
Passing Interface) installed in your machine. Currently tested
MPI libraries include [Open MPI](https://www.open-mpi.org)
and [mpich](https://www.mpich.org). If MPI is not found,
the PySPOD will still work, but in serial.

PySPOD is conveniently available as part of _pip_.
To install the library you can type:

```shell
pip install pyspod
```

We run periodic tests on [GitHub Actions](https://github.com/mathe-lab/PySPOD/actions)
to make sure the library is fully functional.

- To learn what SPOD is and for what is useful for go to [About](./about).

- To learn how to setup a run with PySPOD, go to [Tutorials](./tutorials.html).

- To checkout the team developing PySPOD, go to [Team](./team).

> Please, refer to our [GitHub repository](https://github.com/mathe-lab/PySPOD)
for additional information and latest updates.
