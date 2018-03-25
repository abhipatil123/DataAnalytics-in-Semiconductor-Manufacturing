import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt



def sq_exponential(x, y, l):
    """
    Squared exponential covariance function.

    Parameters
    ----------
    x : Array-like
        First array of coordinates
    y : Array-like
        Second array of coordinates
    l : int or float
        Length-scale value

    Returns
    -------
    Covariance matrix

    """
    d = spatial.distance_matrix(x, y)
    k = np.exp(-(d ** 2) / (2 * l * l))
    return k


def exponential(x, y, l):
    """
    Exponential covariance function.

    Parameters
    ----------
     x : Array-like
        First array of coordinates
    y : Array-like
        Second array of coordinates
    l : int or float
        Length-scale value

    Returns
    -------
    Covariance matrix

    """
    d = spatial.distance_matrix(x, y)
    k = np.exp(-d/l)
    return k


def make_grid(bounding_box, ncell):
    """

    Parameters
    ----------
    bounding_box : List or tuple
        The bounding box to make a grid within. Must include four elements.
    ncell : int
        The number of cells to create along one dimension (if you want a 10x10 grid, pass 10 to this parameter)

    Returns
    -------
    NDArray

    """
    xmax, xmin, ymax, ymin = bounding_box
    xgrid = np.linspace(xmin, xmax, ncell)
    ygrid = np.linspace(ymin, ymax, ncell)
    mX, mY = np.meshgrid(xgrid, ygrid)
    ngridX = mX.reshape(ncell*ncell, 1);
    ngridY = mY.reshape(ncell*ncell, 1);
    return np.concatenate((ngridX, ngridY), axis=1)


def cross_validate(train, l_values, sigma_values, rmse_opt, k_folds, cov_funcs=False, verbose=False):
    """
    K-fold cross validation for the Gaussian prediction model.

    Parameters
    ----------
    train : Array-like
        Training dataset to use for cross-validation
    l_values : Array-like or numeric
        Array of values of l (length-scale) to try, or single value
    sigma_values : Array-like or numeric
        Array of values of sigma (noise) to try, or single value
    rmse_opt : int or float
        Starting value for root mean squared error (RMSE)
    k_folds : int
        Number of folds to use for cross-validation
    cov_funcs : bool, optional
        Tests multiple covariance functions if True (for now, only basic exponential function)
    verbose : bool, optional
        If True, prints RMSE for each combination of values

    Returns
    -------
    l_opt, sigma_opt, cov_func_final, rmse_opt

    """
    if not isinstance(l_values, (list, tuple, np.ndarray, np.array)):
        l_values = list(l_values)

    if not isinstance(sigma_values, (list, tuple, np.ndarray, np.array)):
        sigma_values = list(sigma_values)

    functions = [sq_exponential]
    if cov_funcs:
        functions.append(exponential)

    for f in functions:
        for l in l_values:
            for sigma in sigma_values:
                for k in range(k_folds):

                    folds = np.array_split(train, k_folds)
                    testing = folds.pop(k)
                    training = np.concatenate(folds)

                    krig = SimpleKriging(training_data=training)
                    pred = krig.predict(test_data=testing[:, :2], l=l, sigma=sigma)
                    rmse = (((pred - testing[:, -1:])**2)**.5).mean()

                    if k == 0:
                        local_error = rmse
                    else:
                        if rmse < local_error:
                            local_error = rmse

                    if rmse < rmse_opt:
                        rmse_opt = rmse
                        l_final = l
                        sigma_final = sigma
                        cov_func_final = f

                if verbose:
                    print "l={}, sigma={}, rmse={}".format(l, sigma, local_error)

    return l_final, sigma_final, cov_func_final, rmse_opt


class SimpleKriging(object):
    """
    Model object for predicting and simulating values in two dimensions based on a training dataset based on
    the simple Kriging method (using an underlying Gaussian process).
    """

    def __init__(self, training_data):
        self.training_data = training_data
        self.X = training_data[:, :-1]
        self.Y = training_data[:, -1:]

    def predict(self, test_data, l, sigma, indices=False, cov_function=sq_exponential):
        """

        Parameters
        ----------
        test_data : Array-like
            Set of coordinates to predict values for
        l : int or float
            Length-scale value
        sigma : int or float
            Noise parameter
        indices : bool, optional
            If True, returns results with coordinates.
        cov_function : function, optional
            Uses squared exponential by default. Pass a different covariance function here if desired.

        Returns
        -------

        """
        K_xtest_xtest = cov_function(test_data, test_data, l)
        
        K_xtest_x = cov_function(test_data, self.X, l)
        
        # Get cholesky decomposition (square root) of the
        # covariance matrix
        #L = np.linalg.cholesky(K_xtest_xtest + 0.001*np.eye(len(test_data)))
        #f_prior = np.dot(L, np.random.normal(size=(len(test_data),3)))
        #plt.plot(test_data, f_prior)
            
        #plt.show()
        
        K = cov_function(self.X, self.X, l)
        
        sigma_sq_I = sigma**2 * np.eye(len(self.X))
        inv = np.linalg.inv(K + sigma_sq_I)

        predictions = K_xtest_x.dot(inv).dot(self.Y)

        if indices:
            return np.concatenate((test_data, predictions), axis=1)
        else:
            return predictions
    
    def simulate1(self, bbox, ncells, l, sigma, gamma=0.001, indices=False, cov_function=sq_exponential,
                 show_visual=False, save_kml=None):
        """

        Parameters
        ----------
        bbox : Array-like
            Bounding box of coordinates to create grid within. Must contain four values.
        ncells : int
            The number of cells to create along one dimension (if you want a 10x10 grid, pass 10 to this parameter)
        l : int or float
            Length-scale value
        sigma : int or float
            Noise parameter
        gamma : int or float, optional
            Value for numerical stabilization to add to Cholesky decomposition
        indices : bool, optional
            If True, returns results with coordinates.
        cov_function : function, optional
            Uses squared exponential by default. Pass a different covariance function here if desired.
        show_visual : bool, optional
            Displays visualized results if True
        save_kml : str, optional
            Saves KML file to specified filename (without extension) if this argument is passed.
        """

        grid = make_grid(bounding_box=bbox, ncell=ncells)

        prediction = self.predict(test_data=grid, l=l, sigma=sigma, indices=True)

        K = cov_function(grid, grid, l)
        L = np.linalg.cholesky(K + gamma * np.eye(len(K)))
        u = np.random.normal(size=len(L))

        result = prediction[:, -1] + L.dot(u)

        if show_visual:
            y = grid[:, 0]
            x = grid[:, 1]
            plt.scatter(x, y, c=result)
            plt.colorbar(ticks=[np.min(result), np.max(result)], label='Probe measurement value')
            plt.show()

        if indices:
            return np.concatenate((grid, result[:, None]), axis=1)
        else:
            return result
    
        
