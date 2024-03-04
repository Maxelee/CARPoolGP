"""
This file is where the emulator API is generated. We want the user to be able to do something like the following:

simulation_data, simulation_parameters = Simulations.read_data(file)
desired_parameters = array(values)

Emu = Emulator(simulation_data, simulation_parameters)
Emu.train(**kwargs)

prediction, covariance = Emu.predict(desired_parameters)
"""
import numpy as np
import jax.numpy as jnp
import optax
import jax
jax.config.update("jax_enable_x64", True)
from src import CARPoolProcess


class Emulator:
    def __init__(self, 
                 Simulations, 
                 Surrogates):
        
        self.Simulations = Simulations
        self.Surrogates = Surrogates
        self.param_dimensions = self.Simulations.parameter_dimensions
        self.Surrogate_locs = np.unique(Surrogates.parameters, axis=0).reshape(-1, self.param_dimensions)
        self.params = None
        self.losses = []
    

    def train(self, params=None, learning_rate=0.0, max_iterations=1000, threshold=8):
        """
        Trains the emulator.

        Parameters
        ----------
        params : dict
        **kwargs : dict
            Keyword arguments for the emulator.
        """
        # TODO: Implement break if params is None
        if params is None:
            if self.params is None:
                params = {"log_scaleV"   :0.1*np.ones(self.param_dimensions),
                            "log_ampV"   :1.0*np.ones(self.param_dimensions),
                            "log_scaleW" :1.0*np.ones(self.param_dimensions),
                            "log_ampW"   :1.0*np.ones(self.param_dimensions),
                            "log_scaleX" :1.0*np.ones(self.param_dimensions),
                            "log_ampX"   :1.0*np.ones(self.param_dimensions),                
                            "log_scaleM" :0.0*np.ones(self.param_dimensions),
                            "log_ampM"   :0.0*np.ones(self.param_dimensions),
                            "log_deltaP" :0.0*np.ones(self.param_dimensions),
                            "log_jitterV":5.0,
                            "log_jitterW":5.0,
                            "log_mean"   :0.0}
    
            else:
                params = self.params
        self.threshold=threshold
        # opt = optax.sgd(learning_rate=learning_rate, momentum=0.9)#, nesterov=True)
        opt = optax.amsgrad(learning_rate=learning_rate)
        # opt = optax.adamax(learning_rate=learning_rate)
        opt_state = opt.init(params)
        try:
            self.param_evolution[0]
            self.losses[0]
        except:  
            self.param_evolution = []
            self.losses = [np.inf]
        Y = jnp.concatenate([self.Simulations.quantities, self.Surrogates.quantities])
        for i in range(max_iterations):
            loss, grads = CARPoolProcess.loss(params, 
                                                  jnp.array(self.Simulations.parameters), 
                                                  jnp.array(self.Surrogates.parameters), 
                                                  Y, threshold=self.threshold)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            self.param_evolution.append(params)
            # self.check_params(params) # TODO: Implement check_params
            self.losses.append(loss)
            # if np.abs(np.diff(self.losses)[-1]) < 1e-3:
            #     print('Converged')
            #     break
        self.params = params
        return params

    def predict(self, desired_parameters):
        """
        Predicts the emulator.

        Parameters
        ----------
        desired_parameters : array
            Desired parameters for the emulator.

        Returns
        -------
        prediction : array
            Prediction of the emulator.
        covariance : array
            Covariance of the prediction.
        """
        if self.params is None:
            raise ValueError("Emulator has not been trained.")
        desired_parameters = np.atleast_2d(desired_parameters).reshape(-1, self.param_dimensions)
        
        cov = CARPoolProcess.build_CARPoolCov(self.params, 
                                              self.Simulations.parameters, 
                                              self.Surrogates.parameters, threshold=self.threshold) 
        pred_cov = CARPoolProcess.build_CARPoolCov(self.params, 
                                                   np.concatenate((desired_parameters,
                                                                    self.Simulations.parameters)), 
                                                    self.Surrogates.parameters, 
                                                    noise=None, threshold=self.threshold)
        Y = jnp.concatenate([self.Simulations.quantities, self.Surrogates.quantities])
        prediction, covariance = CARPoolProcess.predict(Y, 
                                                        cov, 
                                                        pred_cov, 
                                                        self.params["log_mean"])
        return prediction, covariance   
    
        
    
    def chi2(self, model, data, cov):
        """
        Calculates the chi2 of the emulator.

        Parameters
        ----------
        model : array
            Model values.
        data : array
            Data values.
        covariance : array
            Covariance of the data.

        Returns
        -------
        chi2 : float
            Chi2 of the emulator.
        """
        return np.sum((data - model)**2/ np.sqrt(np.diag(cov)))

class ActiveLearning(Emulator):

    def __init__(self, Simulations, Surrogates, test_locs, lbs, ubs):
        super().__init__(Simulations, Surrogates)
        self.test_locs = test_locs
        self.lb = lbs
        self.ub = ubs


    def active_learning_step(self, num_new=20, Ngrid=6,normalize=True):

        # Iterate over the following N times:
        for n in range(num_new):
            # First we want to make a grid of parameter values
            theta_grid = self.generate_theta(Ngrid, np.random.randint(0, 10000), normalize=normalize)
            test_stat = np.ones(len(theta_grid))
            
            # Now we iterate through each of the parameters in the grid, and test them 
            for ii, test_theta in enumerate(theta_grid):
                test_stat[ii] = self._test_point(test_theta)                

            # Now we find the minimum point in this test_stat and add to list of next sims
            next_theta = theta_grid[np.argmin(test_stat)]
            next_data = self.predict(next_theta)[0]
            next_surrogate = self.find_nearest_island(next_theta).reshape(next_theta.shape)
            next_surrogate_data = self.predict(next_surrogate)[0]

            # Update the parameters
            self.Simulations.update(next_theta, next_data)
            self.Surrogates.update(next_surrogate, next_surrogate_data)

        next_thetas = self.Simulations.pop(index=-num_new, return_popped=True)[0]
        next_surrogates = self.Surrogates.pop(index=-num_new, return_popped=True)[0]
        
        return next_thetas, next_surrogates
    
    def _test_point(self, test_theta):
        test_theta = np.atleast_1d(test_theta)
        # We need to predict the value of this point in parameter space
        test_data, _ = self.predict(test_theta)
        self.Simulations.update(test_theta, test_data)
        
        # We also want to put a  surrogate at the nearest location
        surrogate_theta_test = self.find_nearest_island(test_theta).reshape(-1)
        surrogate_test_data, _ = self.predict(surrogate_theta_test)
        self.Surrogates.update(surrogate_theta_test, surrogate_test_data)

        # Now we test at our test points
        _, pred_varCP = self.predict(self.test_locs)

        # Remove the points we just added from the class
        self.Simulations.pop()
        self.Surrogates.pop()
        return np.sum(np.abs(np.diag(pred_varCP)))

    def generate_theta(self, N, seed, normalize=True):
        """
        Generates a grid of parameter values.

        Parameters
        ----------
        N : int
            Number of points in the grid.
        seed : int
            Seed for the random number generator.

        Returns
        -------
        theta_grid : array
            Grid of parameter values.
        """
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=self.param_dimensions, scramble=True, seed=seed)
        sample = sampler.random_base2(m=N)
        theta = qmc.scale(sample, self.lb, self.ub)
        if normalize:
            theta = (theta - self.lb) / (self.lb - self.ub) + 1
        return theta
    
    def find_nearest_island(self, parameters):
        """
        Finds the nearest island in the parameter space.

        Parameters
        ----------
        parameters : array
            Parameters to find the nearest island for.

        Returns
        -------
        nearest_island : array
            Nearest island in the parameter space.
        """
        parameters = parameters.reshape(-1, self.param_dimensions)
        nearest_island = np.zeros_like(parameters)
        for i, pi in enumerate(parameters):
            if self.param_dimensions>1:
                nearest_island[i] = self.Surrogate_locs[np.argmin(np.sum((pi[:-1] - self.Surrogate_locs[:, :-1])**2, axis=1))] # TODO: find a way to remove last axis for mass etc. 
            else:
                nearest_island[i] = self.Surrogate_locs[np.argmin((pi - self.Surrogate_locs)**2)] 
        return nearest_island


