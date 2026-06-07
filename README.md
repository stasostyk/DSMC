# DSMC Particle Simulator


![Plot](vis.png)

This project is a 3D CUDA-accelerated implementation of Direct Simulation Monte Carlo (DSMC), which is a stochastic solution of the Boltzmann equation for rarefied gases. These gases are common in supersonic and hypersonic flows where a Knudsen number larger than 1, where Navier Stokes equations have proven to be inaccurate. For example, it is used in aerodynamics to model Space Shuttle re-entry into the atmosphere. 

It works by simulating particles (which correspond to a large number of real particles via the statistical weight parameter) which are in a mesh-free domain and in a flow. Then we group them in cells that are approximately as wide as the mean free path of the molecules, which is a property of rarefied gases that DSMC explots. Within each cell we probabilistically collide particle paris based on a probability derived from the kinetic theory of gases. After enough steps have passed for a flow pattern to emerge, we begin with statistical sampling (Monte Carlo method) to find the average pressure, temperature, and velocity magnitude across the domain.



We began with a hard sphere model for particles in 2D using the No Time Counter scheme. We then extended the project to 3D, implemented a Variable Hard Sphere model for particles, and included the Half-Split-Shuffle algorithm to handle heavy-cells. 

Initial 2D serial version adapted from DSMC lectures found online of [University of Alabama](https://volkov.eng.ua.edu/ME591_491_NEGD/2017-Spring-NEGD-06-DSMC.pdf) and [Purdue University](www.youtube.com/watch?v=cSFr8MTr30Y).

## Building and running

To build:
```
mkdir build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=70 # or higher
make
./DSMC_single_gpu # Avoids MPI
```

With `plot.py` the plot can be generated:
```
./plot.py build/fields_avg.dat output_plot.png
```

When running `./DSMC`, the `.vpi` and `.vpt` files are created that can be opened with ParaView.

### CUDA version

To run CUDA version, follow the steps from [cuda/colab_run.ipynb](cuda/colab_run.ipynb). We have used it to run the code inside Google Colab, using T4 GPUs.
