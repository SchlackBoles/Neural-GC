import numpy as np
from scipy.integrate import odeint


def make_var_stationary(beta, radius=0.97):
    '''Rescale coefficients of VAR model to make stable.'''
    p = beta.shape[0]
    lag = beta.shape[1] // p
    bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
    beta_tilde = np.vstack((beta, bottom))
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))
    nonstationary = max_eig > radius
    if nonstationary:
        return make_var_stationary(0.95 * beta, radius)
    else:
        return beta

def simulate_var(p, T, lag, sparsity=0.2, beta_range=(-0.3, 0.3), sd=0.1, seed=0, zeroing_prob=0.5):
    if seed is not None:
        np.random.seed(seed)

    # Set up Granger causality ground truth.
    GC = np.eye(p, dtype=int)

    # Generate the beta matrix for the VAR process (with lags)
    beta = np.zeros((p, p * lag))

    for i in range(p):
        # Ensure self-dependency for all lags
        for j in range(lag):
            beta[i, i + j * p] = np.random.uniform(beta_range[0], beta_range[1])  # Self-interaction
        
        # Select other random variables that influence variable i
        num_nonzero = int(p * sparsity)  # This determines how many other variables influence i
        if num_nonzero > 0:
            choice = np.random.choice([x for x in range(p) if x != i], size=num_nonzero, replace=False)
            for j in range(lag):
                # Randomly decide whether to zero out the coefficient
                if np.random.rand() > zeroing_prob:  # Keep with probability (1 - zeroing_prob)
                    beta[i, choice + j * p] = np.random.uniform(beta_range[0], beta_range[1], size=num_nonzero)
                    GC[i, choice] = 1  # Update Granger causality matrix

    beta = make_var_stationary(beta)



    # Generate data
    burn_in = 100
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))
    X = np.zeros((p, T + burn_in))
    X[:, :lag] = errors[:, :lag]
    for t in range(lag, T + burn_in):
        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
        X[:, t] += errors[:, t]

    return X.T[burn_in:], beta, GC

def lorenz(x, t, F):
    '''Partial derivatives for Lorenz-96 ODE.'''
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i+1) % p] - x[(i-2) % p]) * x[(i-1) % p] - x[i] + F

    return dxdt


def simulate_lorenz_96(p, T, F=10.0, delta_t=0.1, sd=0.1, burn_in=1000,
                       seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(lorenz, x0, t, args=(F,))
    X += np.random.normal(scale=sd, size=(T + burn_in, p))

    # Set up Granger causality ground truth.
    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    return X[burn_in:], GC
