import pde
import numpy as np
import matplotlib.pyplot as plt

def solve_diffusion(xlim, ylim):
    assert isinstance(xlim, int)
    assert isinstance(ylim, int)
    rows = ylim * 64
    cols = xlim * 64
    grid = pde.CartesianGrid(([0, ylim], [0, xlim]), (rows, cols))
    state = pde.ScalarField.random_uniform(grid)
    #bcs = [{"value": "sin(y)"}, {"value": "sin(x)"}]
    #bcs = [{'value': np.zeros(rows)}, {'value': np.zeros(cols)}]

    diffusivity = "1.01 + tanh(x) * tanh(y)"
    term_1 = f"({diffusivity}) * laplace(c)"
    term_2 = f"dot(gradient({diffusivity}), gradient(c))"
    eq = pde.PDE({"c": f"{term_1} + {term_2}"}, bc={"value": 0})

    storage = pde.MemoryStorage()
    solver = pde.ScipySolver(eq)
    controller = pde.Controller(solver, t_range=0.1, tracker=['progress', storage.tracker(1)])
    result = controller.run(state)

    return diffusivity, storage, result

diff, storage, result = solve_diffusion(1, 1)

print(result.data)
print(storage[0]._data_full.shape)
print(result._data_full.shape)
plt.imshow(result.data)
plt.savefig('result.png')
