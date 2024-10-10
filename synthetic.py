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

def apply_shock(X, shock_magnitude, shock_time, shocked_vars):
    """
    Apply a shock to the system at a specific time step.

    X: np.array, shape (p, T)
        The data matrix containing the time series.
    shock_magnitude: float
        The magnitude of the shock to apply.
    shock_time: int
        The time step at which to apply the shock.
    shocked_vars: list of int
        The indices of the variables that are shocked.
    """
    X_shocked = X.copy()
    
    for var in shocked_vars:
        X_shocked[var, shock_time] += shock_magnitude
    
    return X_shocked

def propagate_shock(X, beta, errors, t_shock, lag):
    """
    Propagates a shock in a given series from time step t_shock onwards.
    
    Parameters:
    X : np.ndarray
        The original time series data (p, T) where p is the number of variables and T is the number of time steps.
    beta : np.ndarray
        The coefficient matrix for the VAR process.
    errors : np.ndarray
        The noise/error matrix added to the series.
    t_shock : int
        The time step at which the shock occurred.
    lag : int
        The number of lags in the VAR process.
    
    Returns:
    np.ndarray
        The series after shock propagation.
    """
    p, T = X.shape  


    for t in range(t_shock + 1, T):
        # Compute the next time step based on the previous lagged values and VAR process
        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
        # Add the corresponding error term for this time step
        X[:, t] += errors[:, t]
    
    return X



def calculate_irfs(A1, A2, steps):
    """
    Calculate the Impulse Response Functions for a VAR(2) model as a 3D numpy array.

    Parameters:
    A1 : np.ndarray
        The autoregressive coefficient matrix for lag 1.
    A2 : np.ndarray
        The autoregressive coefficient matrix for lag 2.
    steps : int
        The number of time steps to compute IRFs for.

    Returns:
    np.ndarray
        A 3D array of IRF matrices from Phi_0 to Phi_steps.
    """
    k = A1.shape[0]  # Assuming A1 and A2 are square matrices of the same size
    irfs = [np.eye(k), A1]  # Initialize with Phi_0 as identity and Phi_1 as A1

    # Start computing from Phi_2
    for i in range(2, steps):
        # Phi_i = Phi_{i-1} * A1 + Phi_{i-2} * A2
        Phi_i = np.dot(irfs[i-1], A1) + np.dot(irfs[i-2], A2)
        irfs.append(Phi_i)

    # Convert the list of matrices to a 3D numpy array
    return np.array(irfs)