import numpy as np

def f(point):
    x, y = point
    return x**2 + x*y + y**2 - 6*x - 9*y

def nelder_mead(x0, alpha=1, beta=0.5, gamma=2, max_iter=100):

    n = len(x0)
    simplex = np.zeros((n+1, n))
    simplex[0] = x0

    for i in range(n):
        x = np.copy(x0)
        x[i] += 1
        simplex[i+1] = x

    for i in range(max_iter):
        values = np.array([f(x) for x in simplex])
        order = np.argsort(values)
        best, good, worst = order[0], order[1], order[-1]
        centroid = np.mean(simplex[order[:-1]], axis=0)
        reflection = centroid + alpha*(centroid-simplex[worst])
        if f(reflection) < values[good]:
            expansion = centroid + gamma*(reflection-centroid)
            if f(expansion) < f(reflection):
                simplex[worst] = expansion
            else:
                simplex[worst] = reflection
        else:
            if f(reflection) < values[worst]:
                simplex[worst] = reflection
            contraction = centroid + beta*(simplex[worst]-centroid)
            if f(contraction) < values[worst]:
                simplex[worst] = contraction
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[best] + 0.5*(simplex[i]-simplex[best])
    return simplex[best]
x0=[0,0]
print(nelder_mead(x0))