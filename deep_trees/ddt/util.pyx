import numpy as np

def cache(name):
    def wraps(f):
        def func(self, *args):
            if not hasattr(self, "_cache"):
                self._cache = {}
            if name not in self._cache:
                self._cache[name] = f(self, *args)
            return self._cache[name]
        return func
    return wraps

def make_divergence(float c):
    def a(float t):
        return c / (1 - t)
    def A(float t):
        return -c * np.log(1 - t)
    return a, A

def make_harmonic():
    cache = {1 : 1.0}
    def harmonic(int a):
        if a not in cache:
            cache[a] = 1.0 / a + harmonic(a - 1)
        return cache[a]
    return harmonic
harmonic = make_harmonic()

def make_log_factorial():
    cache = {0 : 0.0}
    def log_factorial(int a):
        if a not in cache:
            cache[a] = np.log(a) + log_factorial(a - 1)
        return cache[a]
    return log_factorial
log_factorial = make_log_factorial()

def make_log_normal(D):
    const = D / 2.0 * np.log(2 * np.pi)

    def log_normal(x, mu, sigma):
        _, logdet = np.linalg.slogdet(sigma)
        pre = -(const + 1/2.0 * logdet)
        delta = x - mu
        return pre + -0.5 * (np.dot(delta, np.dot(np.linalg.inv(sigma), delta)))
    return log_normal
