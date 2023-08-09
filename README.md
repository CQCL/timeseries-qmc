# TimeseriesQMC

This is a python library for running and testing the time-seriers quantum Monte Carlo algorithm of model Hamiltonians.
Time-series quantum Monte Carlo is a hybrid classical-quantum algorithm for evaluting finite-temperature obervables 
by sampling easily preparable states and estimating their thermal weights from real-time dynamics simulated on a quantum
computer.

This library was developed by the condesend matter group of [Quantinuum](https://www.quantinuum.com/).

## Getting started

This library is available for Python>=3.8 on Linux and MacOS.
To install it, run:

```shell
pip install timeseries-qmc
```

This automatically installs all the dependecies except for [QuSpin](https://quspin.github.io/QuSpin/index.html).
For instructions on the best practice for installing QuSpin alongside other pip packages check our 
[installation guide](https://cqcl.github.io/timeseries-qmc/build/html/install.html).


Full documentation of the **timeseries-qmc** package can be found 
[here](https://cqcl.github.io/timeseries-qmc/build/html/index.html).
It includes an [API reference](https://cqcl.github.io/timeseries-qmc/build/html/api.html), 
[examples](https://cqcl.github.io/timeseries-qmc/build/html/examples.html) and a 
[tutorial](https://cqcl.github.io/timeseries-qmc/build/html/tutorial.html) for a quick introduction on how to use the library.

## How to cite

If you use this library for a work published in an academic journal, we apperciate citing 
[this paper](https://arxiv.org/abs/2305.19322) to acknowledge the effort put into the development.

## Acknowledgment

This work was supported by the German Federal Ministry of Education and Research (BMBF) through the project 
[EQUAHUMO](https://www.quantentechnologien.de/forschung/foerderung/anwendungsnetzwerk-fuer-das-quantencomputing/equahumo.html)
(grant number 13N16069) within the funding program quantum technologies - from basic research to market.


## License

The code is licensed under [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt).
