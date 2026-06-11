from typing import Optional
from numpy.typing import NDArray
import numpy as np 
from scipy.stats import qmc

def LatinHyperSampling(n: int, l_bounds: list[float], u_bounds: list[float], seed: Optional[int] = None) -> NDArray[np.float64]:
    """This function returns the subset of parameter values from the complete set of parameter-space using Latin Hypercube Sampling

    Args:
        n (int): Number of models or samples we want to run from our models of complete parameter-space
        l_bounds (list[float]): scaling the axes of Latin Hypercube as per the parameter ranges - lower bound
        u_bounds (list[float]): scaling the axes of Latin Hypercube as per the parameter ranges - upper bound
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        NDArray[np.float64]: Array of shape (n, d) containing sampled parameter combinations.
    """
    d: int = len(l_bounds)            # Dimensionality of our parameter space. For damping wings project, it is 4, namely: x_HI, M_min, t_q, M_qso
    lhs_sampler:qmc.LatinHypercube    # Latin Hypercube dimension - depending upon the number of parameters
    samples: NDArray[np.float64]      # Sampling/ distributing "n" points throughout the Latin Hypercube - n points in [0,1]^4
    params: NDArray[np.float64]
    
    
    lhs_sampler = qmc.LatinHypercube(d)
    samples = lhs_sampler.random(n)

    params = qmc.scale(samples, l_bounds, u_bounds)
    
    return params
    