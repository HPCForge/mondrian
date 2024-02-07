# ASM Sample

This is a sample experiment, applying the Alternating Schwarz Method with different subdomain solvers:
a finite difference solver and a simple neural network.

The model problem is just the poisson equation with dirichlet boundaries: `div(grad(u)) = 1`, with u = g on the boundary.

The main purpose of this is to illustrate
 1. the limitations of ASM as a solver,
 2. the limitations of using a neural network naively as a subdomain solver,
 3. some simple analysis on what is going wrong.
