"""
This code performs gaussian process regression given some set of data. It interacts with the CARPoolKernels classes to do this and tinygp.

The minimization routine is done with jax which we believe to be optimal. 
"""
import jax
import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as linalg
jax.config.update("jax_enable_x64", True)
from CARPoolGP import CARPoolKernels

@jax.jit
@jax.value_and_grad
def loss(params, theta, surrogate_theta, Y, threshold):
    """
    Return the loss and gradient of the loss for gradient descent
    """
    cov = build_CARPoolCov(params, theta, surrogate_theta, threshold=threshold)
    
    # Compute liklihood
    alpha, scale_tril = decomp(cov, Y, params["log_mean"])
    L = log_liklihood(scale_tril, alpha)

    return -L


@jax.jit
@jax.value_and_grad
def loss_nocorr(params, theta, surrogate_theta, Y, threshold):
    """
    Return the loss and gradient of the loss for gradient descent
    """
    cov = build_CARPoolCov_nocorr(params, theta, surrogate_theta, threshold=threshold)
    
    # Compute liklihood
    alpha, scale_tril = decomp(cov, Y, params["log_mean"])
    L = log_liklihood(scale_tril, alpha)

    return -L

#@jax.jit
#def build_CARPoolCov_nocorr(params, theta, surrogate_theta, noise=0, threshold=8):
#    N_theta     = len(theta)
#    N_surrogates = len(surrogate_theta)
#    ampV = jnp.exp(params["log_ampV"])
#    scaleV = jnp.exp(params["log_scaleV"])
#
#    # Build Kernels with current parameter values
#    Vkernel = CARPoolKernels.EKernel(scaleV, ampV)
#    
#    V       = Vkernel(theta, theta)
#    W       = Vkernel(surrogate_theta, surrogate_theta)
#    X       = Vkernel(theta, surrogate_theta)
#    
#    C       = jnp.block([[V, X],[X.T, W]])
#    
#    if noise is None:
#        return C
#    
#    # Build the noise fluctutaions
#    IsigmaV = jnp.exp(params["log_jitterV"])**2 * jnp.eye(N_theta)
#    IsigmaW = jnp.exp(params["log_jitterV"])**2 * jnp.eye(N_surrogates)
#    M = jnp.zeros((N_theta, N_surrogates))
#    noise = jnp.block([[IsigmaV, M], [M.T, IsigmaW]])
#
#    cov   = C + noise
#    return cov

#@jax.jit
#def build_CARPoolCov_nocorr(params, theta, surrogate_theta, noise=0, threshold=8):
#    N_theta = len(theta)
#    N_surrogates = len(surrogate_theta)
#    t = jnp.concatenate((theta, surrogate_theta))
#    
#    # Separate scales for physics vs mass
#    #physics_scaleV = jnp.exp(params["log_scaleV"][:-1])  # First N-1 parameters
#    #mass_scaleV = jnp.exp(params["log_scaleV"][-1])      # Last parameter (mass)
#    
#    #physics_ampV = jnp.exp(params["log_ampV"][:-1])
#    #mass_ampV = jnp.exp(params["log_ampV"][-1])
#    
#    # Use Matern kernel instead of RBF
##    Vkernel = CARPoolKernels.MaternKernel(physics_scaleV, physics_ampV, nu=2.5)
#    # Or use composite kernel
##    Vkernel = CARPoolKernels.CompositeKernel(physics_scaleV, mass_scaleV, physics_ampV, mass_ampV)
##    Vkernel = CARPoolKernels.AdditiveCompositeKernel(physics_scaleV, mass_scaleV, physics_ampV, mass_ampV)
#    Vkernel = CARPoolKernels.ARDKernel(jnp.exp(params['log_scaleV']), jnp.exp(params['log_ampV']))
#    
#    C = Vkernel(t, t)
#    
#    if noise is None:
#        return C
#    
#    # Heteroscedastic noise (different noise for different mass ranges)
#    base_jitter = jnp.exp(params["log_jitterV"])**2
#    
#    # Optional: mass-dependent noise
#    # mass_values = t[:, -1]  # Assuming mass is last column
#    # noise_scale = 1.0 + 0.1 * mass_values  # Example: more noise at higher masses
#    # IsigmaV = base_jitter * jnp.diag(noise_scale)
#    
#    IsigmaV = base_jitter * jnp.eye(N_theta + N_surrogates)
#    cov = C + IsigmaV
#    return cov

@jax.jit
def build_CARPoolCov_nocorr(params, theta, surrogate_theta, noise=0, threshold=8):
    N_theta     = len(theta)
    N_surrogates = len(surrogate_theta)
    t = jnp.concatenate((theta, surrogate_theta))
    scaleV = jnp.exp(params["log_scaleV"])
    ampV = jnp.exp(params["log_ampV"])

    # Build Kernels with current parameter values
    #Vkernel = CARPoolKernels.WKernel(scaleV, jnp.exp(params["log_ampV"]), scaleV2, jnp.exp(params["log_ampV2"]))
    Vkernel = CARPoolKernels.VWKernel(scaleV, ampV) 
    #Wkernel = CARPoolKernels.VWKernel(scaleV, ampV)
    #Xkernel = CARPoolKernels.VWKernel(scaleV, ampV)
    
    C       = Vkernel(t, t)
    #W       = Vkernel(surrogate_theta, surrogate_theta)
    #X       = Vkernel(theta, surrogate_theta)
    
    #C       = jnp.block([[V, X],[X.T, W]])
    
    if noise is None:
        return C
    
    # Build the noise fluctutaions
    IsigmaV = jnp.exp(params["log_jitterV"])**2 * jnp.eye(N_theta + N_surrogates)
    #IsigmaW = jnp.exp(params["log_jitterV"])**2 * jnp.eye(N_surrogates)
    #M = jnp.zeros((N_theta, N_surrogates))
    #noise = jnp.block([[IsigmaV, M], [M.T, IsigmaW]])

    cov   = C + IsigmaV
    return cov

@jax.jit
def build_CARPoolCov(params, theta, surrogate_theta, noise=0, threshold=8):
    N_theta     = len(theta)
    N_surrogates = len(surrogate_theta)
    
    scaleV = params["log_scaleV"]
    scaleV2 = params["log_scaleV2"]

    scaleM = params["log_scaleM"]

    # Build Kernels with current parameter values
    Vkernel = CARPoolKernels.WKernel(scaleV, jnp.exp(params["log_ampV"]), scaleV2, jnp.exp(params["log_ampV2"]))
    
    V       = Vkernel(theta, theta)
    W       = Vkernel(surrogate_theta, surrogate_theta)
    X       = Vkernel(theta, surrogate_theta)
    
    C       = jnp.block([[V, X],[X.T, W]])
    
    if noise is None:
        return C
    
    # Build the noise fluctutaions
    Mkernel = CARPoolKernels.EKernel(scaleM)
    M       = Mkernel(theta, surrogate_theta) * jnp.exp(params["log_jitterV"])**2  * jnp.eye(N_theta, N_surrogates)
    IsigmaV = jnp.exp(params["log_jitterV"])**2 * jnp.eye(N_theta)
    IsigmaW = jnp.exp(params["log_jitterV"])**2 * jnp.eye(N_surrogates)


    noise = jnp.block([[IsigmaV, M], [M.T, IsigmaW]])
    cov   = C + noise
    return cov

@jax.jit
def sigmoid(x, threshold=8):
    return threshold/(1 + jnp.exp(-x))
def predict(Y, cov, cov_new, mu_y):
    """
    mean = Ks C^{-1} (Y-\mu_Y) + \mu_Y
    cov = Kss - Ks C^{-1}Ks^T

    Ks = covariance of new thetas with old thetas,
    Kss= covariance of new thetas
    C  = covariance from GP
    x  = Value of params
    gp_mean = mean function

    returns mean and cov
    """
    ltn = cov_new.shape[0] - cov.shape[0]
    mean = cov_new[:ltn, ltn:] @ np.linalg.inv(cov)@(Y - mu_y) + mu_y
    cov = cov_new[:ltn, :ltn] - \
        cov_new[:ltn, ltn:] @ np.linalg.inv(cov) @  cov_new[:ltn, ltn:].T
    return mean, cov

@jax.jit
def decomp(cov, Q, mean):
    scale_tril = linalg.cholesky(cov, lower=True)
    alpha = linalg.solve_triangular(scale_tril, Q-mean, lower=True)
    return alpha, scale_tril


@jax.jit
def invdecomp(cov, Q, mean):
    scale_tril = linalg.cholesky(cov, lower=True)
    alpha = linalg.solve_triangular(scale_tril, Q-mean, lower=True, trans=1)
    return alpha, scale_tril


@jax.jit
def log_liklihood(scale_tril, alpha):
    return -0.5 * jnp.sum(jnp.square(alpha)) - \
        jnp.sum(jnp.log(jnp.diag(scale_tril))) + \
        0.5 * scale_tril.shape[0] * jnp.log(2 * jnp.pi)
