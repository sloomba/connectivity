from scipy.stats import gaussian_kde, rv_continuous
from utils import *
    
class GaussianKernelDistribution(rv_continuous):

    def __init__(self, data, gaussian_bw=None, name='empirical_distribution'):
        rv_continuous.__init__(self, name=name)
        self.a = float(min(data))
        self.b = float(max(data))
        self.kernel = gaussian_kde(data, gaussian_bw)
        self.norm = self.kernel.integrate_box_1d(self.a, self.b)

    def _pdf(self, x):
        if isiterable(x):
            output = np.zeros(len(x))
            for i in range(len(x)):
                if self.a <= x[i] <= self.b: output[i] = self.kernel.evaluate([[x[i]]])[0]/self.norm
            return output
        else:
            if self.a <= x <= self.b: return self.kernel.evaluate([[x]])[0]/self.norm
            else: return 0.0

    def _cdf(self, x):
        if isiterable(x):
            output = np.zeros(len(x))
            for i in range(len(x)):
                if self.a < x[i] < self.b: output[i] = self.kernel.integrate_box_1d(self.a, x[i])/self.norm
                elif x[i] >= self.b: output[i] = 1.0
            return output
        else:
            if self.a < x < self.b: return self.kernel.integrate_box_1d(self.a, x)/self.norm
            elif x <= self.a: return 0.0
            else: return 1.0

    def support(self):
        return (self.a, self.b)