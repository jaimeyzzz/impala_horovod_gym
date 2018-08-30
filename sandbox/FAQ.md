Q: `mpirun` running into problem on MAC?

A: It does not support the `-mca btl ^openib` arguments on MAC. 
Change it to `-mca btl ^tcp`.
See the Horovod issue [here](https://github.com/uber/horovod/issues/347).


Q: what dose `global_step` mean in case of multiple learners?
And how to tell the speedup?
A: See the Horovod issues [here](https://github.com/uber/horovod/issues/71) and [here](https://github.com/uber/horovod/issues/461).