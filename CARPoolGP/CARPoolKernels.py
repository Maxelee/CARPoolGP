"""
This code is meant to develop the CARPool kernels that are associated with arxiv:..... 

The kernels interact generally with tinygp (Foreman-Mackey - ). This is the backend we use to write the kernels. 

We need a kernel V, for smooth variation in Q
We need a kernel W, for the smooth variation in R
We need a kernel Isigma, for the noise variations in Q
We need a kernel E, for the designed noise variations in R
We need a kernel X, for the cross correlation in Q and R
We need a kernel M, for the cross correlation in the noise variations between Q and R

We then want a block kernel [[V, X],[X^T, W]] for the smooth variation and 
We then want a block kernel [[Isigma, M], [M^T, E]] for the noise variations
"""
from tinygp import kernels
from tinygp.kernels.distance import Distance, L1Distance, L2Distance
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)


class QKernel(kernels.Kernel):
    """
    Custom kernel for carpool that can take N-dimensional scale. This is realy just a squared 
    exponential kernel
    """
    # def __init__(self, amp, scale):
    #     self.scales = jnp.atleast_1d(scale)
    #     self.amp   = jnp.atleast_1d(amp)
    amp: jax.Array
    l1: jax.Array
    l2: jax.Array
    q: jax.Array
    D: jax.Array
    
    def evaluate(self, X1, X2):
        X1 = X1 / self.l1
        X2 = X2 / self.l2

        x = jnp.atleast_1d(jnp.abs(jnp.sqrt((X2 - X1))))
        j = self.q#jnp.floor(36.0/2) + self.q + 1
        #fmax = jnp.max(0.0, 1.0 - x)**(j + self.q)
        K = jnp.abs((1-x)**(j+2) * (1 + (j+2)*x + (j**2 + 4 * j + 4) / (3) * x**2))
        return jnp.prod(self.amp * K) 


class VWKernel(kernels.Kernel):
    """
    Custom kernel for carpool that can take N-dimensional scale. This is realy just a squared 
    exponential kernel
    """
    # def __init__(self, amp, scale):
    #     self.scales = jnp.atleast_1d(scale)
    #     self.amp   = jnp.atleast_1d(amp)
    scales: jax.Array
    amp: jax.Array

    def evaluate(self, X1, X2):
        x = jnp.atleast_1d(jnp.sqrt((X2 - X1)**2))
        # return jnp.prod(self.amp * jnp.exp(-0.5 * x**2 / self.scale**2))
        return jnp.prod(self.amp * jnp.exp(-0.5 * x**2/self.scales**2))
    
class WKernel(kernels.Kernel):
    """
    Custom kernel for carpool that can take N-dimensional scale. This is realy just a squared 
    exponential kernel
    """
    # def __init__(self, amp, scale):
    #     self.scales = jnp.atleast_1d(scale)
    #     self.amp   = jnp.atleast_1d(amp)
    scales: jax.Array
    amp: jax.Array
    scales2: jax.Array
    amp2: jax.Array

    def evaluate(self, X1, X2):
        x = jnp.atleast_1d(jnp.sqrt((X2 - X1)**2))
        arg = jnp.sqrt(3) + x/self.scales
        return jnp.prod(self.amp* (1 + arg)*jnp.exp(-arg) * self.amp2 * jnp.exp(-0.5 * x**2/self.scales2**2))
    
class XKernel(kernels.Kernel):
    """
    Custom kernel for carpool that can take N-dimensional scale
    """
    # def __init__(self, amp, scale, deltaP):
    #     self.scales   =jnp.atleast_1d(scale)
    #     self.deltaP  =jnp.atleast_1d(deltaP)
    #     self.amp     = jnp.atleast_1d(amp)
    scales: jax.Array
    amp: jax.Array
    deltaP: jax.Array
    
    def evaluate(self, X1, X2):
        x = jnp.atleast_1d(jnp.sqrt((X2 - X1)**2))
        # return jnp.prod(self.amp*jnp.exp(-0.5 * (x**2 + self.deltaP)/self.scale))
        return jnp.prod(self.amp * jnp.exp(-0.5 * (x**2 + self.deltaP)/self.scales**2))
    
# class XKernel(kernels.Kernel):
#     """
#     Custom kernel for carpool that can take N-dimensional scale
#     """
#     def __init__(self, amp, scale, deltaP):
#         self.scale   = jnp.atleast_1d(scale)
#         self.deltaP  = jnp.atleast_1d(deltaP)
#         self.amp     = jnp.atleast_1d(amp)

#     def evaluate(self, X1, X2):
#         x = jnp.atleast_1d(jnp.sqrt((X2 - X1)**2))
#         arg = jnp.sqrt(3) + (x**2 + self.deltaP)/self.scale**2
#         return jnp.prod(self.amp*(1 + arg)*jnp.exp(-arg))
    
class EKernel(kernels.Kernel):
    """
    Custom kernel for carpool that can take N-dimensional scale. This is realy just a linear 
    exponential kernel
    """
    # def __init__(self, scale):
    #     self.scales = jnp.atleast_1d(scale)
    scales: jax.Array
        
    def evaluate(self, X1, X2):
        x = jnp.atleast_1d(jnp.sqrt((X2 - X1)**2))
        return jnp.prod(jnp.exp(-0.5 * x/ self.scales**2))
    
class MaternKernel(kernels.Kernel):
    """
    Matern kernel - more flexible than RBF
    """
    scales: jax.Array
    amp: jax.Array
    nu: float = 2.5  # Smoothness parameter

    def evaluate(self, X1, X2):
        x = jnp.atleast_1d(jnp.sqrt(jnp.sum((X2 - X1)**2 / self.scales**2)))
        
        if self.nu == 0.5:
            # Exponential kernel
            return self.amp * jnp.exp(-x)
        elif self.nu == 1.5:
            # Matern 3/2
            sqrt3_x = jnp.sqrt(3) * x
            return self.amp * (1 + sqrt3_x) * jnp.exp(-sqrt3_x)
        elif self.nu == 2.5:
            # Matern 5/2 - good default
            sqrt5_x = jnp.sqrt(5) * x
            return self.amp * (1 + sqrt5_x + (5 * x**2) / 3) * jnp.exp(-sqrt5_x)
        else:
            # General Matern (more expensive)
            from jax.scipy.special import gamma, kv
            term1 = (2**(1-self.nu)) / gamma(self.nu)
            term2 = (jnp.sqrt(2*self.nu) * x)**self.nu
            term3 = kv(self.nu, jnp.sqrt(2*self.nu) * x)
            return self.amp * term1 * term2 * term3

class CompositeKernel(kernels.Kernel):
    """
    Separate treatment for physics parameters vs mass
    """
    physics_scales: jax.Array  # For first N-1 dimensions
    mass_scale: float          # For last dimension (mass)
    physics_amp: jax.Array
    mass_amp: float
    
    def evaluate(self, X1, X2):
        # Split physics and mass dimensions
        physics_diff = (X2[:-1] - X1[:-1]) / self.physics_scales
        mass_diff = (X2[-1] - X1[-1]) / self.mass_scale
        
        # Matern 5/2 for physics
        physics_dist = jnp.sqrt(jnp.sum(physics_diff**2))
        sqrt5_phys = jnp.sqrt(5) * physics_dist
        physics_kernel = jnp.prod(self.physics_amp) * (1 + sqrt5_phys + (5 * physics_dist**2) / 3) * jnp.exp(-sqrt5_phys)
        
        # RBF for mass (smooth scaling relation)
        mass_kernel = self.mass_amp * jnp.exp(-0.5 * mass_diff**2)
        
        return physics_kernel * mass_kernel

class AdditiveCompositeKernel(kernels.Kernel):
    """
    Additive composite kernel - more numerically stable
    """
    physics_scales: jax.Array
    mass_scale: float
    physics_amp: jax.Array
    mass_amp: float
    
    def evaluate(self, X1, X2):
        # Split dimensions
        physics_diff = (X2[:-1] - X1[:-1]) / self.physics_scales
        mass_diff = (X2[-1] - X1[-1]) / self.mass_scale
        
        # Physics kernel (Matern 5/2)
        physics_dist = jnp.sqrt(jnp.sum(physics_diff**2))
        sqrt5_phys = jnp.sqrt(5) * physics_dist
        physics_kernel = jnp.prod(self.physics_amp) * (1 + sqrt5_phys + (5 * physics_dist**2) / 3) * jnp.exp(-sqrt5_phys)
        
        # Mass kernel (RBF)
        mass_kernel = self.mass_amp * jnp.exp(-0.5 * mass_diff**2)
        
        # ADD instead of multiply - much more stable
        return physics_kernel + mass_kernel

class ARDKernel(kernels.Kernel):
    """
    RBF kernel with different length scales per dimension
    """
    scales: jax.Array  # One scale per dimension
    amp: float         # Single amplitude
    
    def evaluate(self, X1, X2):
        # Weighted distance - each dimension has its own scale
        diff = (X2 - X1) / self.scales
        dist_sq = jnp.sum(diff**2)
        return self.amp * jnp.exp(-0.5 * dist_sq)
