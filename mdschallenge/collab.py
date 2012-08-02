import numpy as np
from scipy.optimize import fmin_cg


def cost(p, Y, R, alpha):
    """
    Calculates collaborative filtering cost function.

    Arguments
    ----------
    P: (m+u)xn feature and weight matrix
    Y: mxu rating matrix
    R: mxu i has been rated by j boolean matrix
    alpha: regularization parameter controls model complexity

    Return
    ----------
    Value of cost function
    """
    m, u = Y.shape
    n = int(p.size / float(m + u))
    X = np.resize(p[:m * n], (m, n))
    Theta = np.resize(p[m * n:], (u, n))

    J = 1 / 2. * np.sum(np.multiply(np.power(np.dot(X, Theta.T) - Y, 2), R))
    J += alpha / 2. * (np.sum(np.power(Theta, 2)) + np.sum(np.power(X, 2)))

    return J


def grad(p, Y, R, alpha):
    """
    Calculates parameter gradients of collaborative filtering cost function.
    The parameters are m feature and u weight vectors.

    Arguments
    ----------
    P: (m+u)xn feature and weight matrix
    Y: mxu rating matrix
    R: mxu i has been rated by j boolean matrix
    alpha: regularization parameter controls model complexity

    Return
    ----------
    Vector of gradients ((m+u)xn feature and weight matrix)
    """
    m, u = Y.shape
    n = p.size / (m + u)
    X = np.resize(p[:m * n], (m, n))
    Theta = np.resize(p[m * n:], (u, n))

    X_grad = np.dot(np.multiply((np.dot(X, Theta.T) - Y), R), Theta) + alpha * X
    Theta_grad = np.dot(np.multiply((np.dot(X, Theta.T) - Y).T, R.T), X) + alpha * Theta

    return np.ravel(np.vstack((X_grad, Theta_grad)))


def fit(Y, R, alpha, n):
    """
    Fits the parameters of the collaborative filtering model

    Arguments
    ----------
    Y: mxu rating matrix
    R: mxu i has been rated by j boolean matrix
    n: Number of features.
    alpha: regularization parameter controls model complexity.

    Return
    ----------
    (X,Theta)
    X: mxn feature matrix
    Theta: uxn weight matrix
    """
    m, u = Y.shape
    p = np.random.random((m + u) * n)

    # minimize cost function
    costf = lambda x: cost(x, Y, R, alpha)
    gradf = lambda x: grad(x, Y, R, alpha)
    p = fmin_cg(costf, p, fprime=gradf, maxiter=100, disp=False)

    # unroll parameters
    X = np.resize(p[:m * n], (m, n))
    Theta = np.resize(p[m * n:], (u, n))

    return (X, Theta)


def predict(X, y, r, alpha):
    """
    Predicts ratings for a new user.
    Uses ridge regression to fit the parameters.

    Arguments
    ----------
    X: mxn feature matrix
    y: m rating vector
    r: rated boolean vector
    alpha: regularization parameter controls model complexity.

    Return
    ----------
    Vector of ratings
    """
    aX = X
    X = X[r, :]
    y = np.array([y[r]]).T
    G = alpha / 2. * np.eye(X.shape[1])
    Theta = np.dot(np.linalg.inv(np.dot(X.T, X) + np.dot(G.T, G)), np.dot(X.T, y))
    return np.dot(aX, Theta)


class CF(object):
    """
    Regression based Collaborative Filtering
    """
    def __init__(self, alpha=10, n=10):
        """
        Arguments
        ----------
        alpha: regularization parameter controls model complexity.
        n: number of features
        """
        self.n = n
        self.alpha = alpha

    def fit(self, Y, R):
        """
        Fits model to rating matrix Y
        Uses ridge regression to fit the parameters of new user.

        Arguments
        ----------
        Y: mxu rating matrix
        R: mxu i has been rated by j boolean matrix
        """
        # Mean normalize ratings using "add one" laplace smoothing.
        # This is accomplished by using a dummy user that has rated all
        # movies with the mean global rating.
        mean = np.mean(np.divide(np.sum(np.multiply(Y, R), 1), np.sum(R, 1)))
        self.means = np.array([np.divide(np.sum(np.multiply(Y, R), 1) + mean,
            np.sum(R, 1) + 1)]).T

        Y = Y - self.means

        # Save model parameters
        self.X, self.Theta = fit(Y, R, self.alpha, self.n)
        return (self.X, self.Theta)

    def predict(self, y, r):
        """
        Fits model to rating matrix Y
        Uses ridge regression to fit the parameters of new user.

        Arguments
        ----------
        y: vector of ratings from new user
        r: i has been rated by new user boolean vector
        """
        # Predict ratings for new user
        ratings = predict(self.X, y, r, self.alpha)

        # Add means to predictions
        return ratings + self.means


if __name__ == '__main__':
    """
    Runs the ml-class.org movie ratings example from exercise 8.
    """
    from scipy.io import loadmat
    # Model parameters
    alpha = 10
    n = 10

    # Load existing ratings and movies
    D = loadmat('ex8_movies.mat')
    Y, R = (D['Y'], D['R'])
    f = open('movie_ids.txt', 'rb')
    movies = np.array([str.strip(l.partition(' ')[-1]) for l in f])

    # New user ratings
    y = np.zeros(Y.shape[0])
    y[0] = 4
    y[96] = 2
    y[6] = 3
    y[11] = 5
    y[53] = 4
    y[63] = 5
    y[65] = 3
    y[68] = 5
    y[182] = 4
    y[225] = 5
    y[354] = 5
    r = y != 0

    # Create model
    cf = CF(alpha, n)
    cf.fit(Y, R)

    # Predict movies for new user
    ratings = cf.predict(y, r)
    print 'Top recommendations for new user:'
    ranks, ids = zip(*sorted(zip(ratings, xrange(len(ratings))), reverse=True))
    for i in xrange(n):
        print 'Predicting rating %.2f for movie %s' % (ranks[i], movies[ids[i]])