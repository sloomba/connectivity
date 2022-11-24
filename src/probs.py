from scipy.stats import gaussian_kde, rv_continuous, rv_discrete
from scipy.special import expi, digamma
from .utils import *
    
class GaussianKernelDistribution(rv_continuous):

    def __init__(self, data, gaussian_bw=None, name='gaussian_kernel_distribution'):
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

class InversePoissonDistribution(rv_continuous):

    def __init__(self, mu=1., limit=1e-2, name='inverse_poisson_distribution'):
        rv_continuous.__init__(self, name=name)
        self.a = 0.
        self.b = 1.
        if mu<0: raise ValueError('parameter "mu" must be non-negative')
        self.mu = float(mu)
        self.lim = float(limit)

    def _pdf(self, k):
        mu = self.mu
        y = self.x2y(k)
        if mu==0: output = np.array(y==1, dtype=float)
        else: output = np.exp((y-1)*np.log(mu) - np.log(self.factorial(y-1)) - mu)
        return output

    def _cdf(self, k):
        mu = self.mu
        y = self.x2y(k)
        if isiterable(y): output = np.array([1-self.pdf(1/np.arange(1, y[i])).sum() for i in range(len(y))])
        else: output = 1-self.pdf(1/np.arange(1, y)).sum()
        return output

    def _munp(self, n, analytic=True):
        mu = self.mu
        if mu==0: return 1.
        if analytic:
            if n==1: return (1-np.exp(-mu))/mu
            elif n==2: return np.exp(-mu)*(expi(mu)-np.log(mu)+digamma(1))/mu
            else: analytic = False
        if not analytic: return sum([self.pdf(1/i)/i**n for i in range(1, int(1/self.lim)+1)])

    def factorial(self, x):
        if isiterable(x): return np.array([self.factorial(i) for i in x])
        else: return float(np.math.factorial(x))

    def x2y(self, x):
        if isiterable(x):
            y = np.array(x)
            y[y<self.lim] = self.lim
        else:
            y = x
            if y<self.lim: y = self.lim
        y = np.ceil(1/y)
        return y

    def support(self):
        return (self.a, self.b)

class ShortestPathDistribution(rv_discrete):

    def __init__(self, psi=1., num=1e3, pi=None, x=None, recursion_lim=1e2, psi_args=(), name='shortest_path_distribution'):
        rv_discrete.__init__(self, name=name, shapes='mean_source, mean_target, gcc_source, gcc_target, apx, init_corrected, w')
        self.psi, self.pi, self.x = self.check_psipi(psi, pi, x, psi_args)
        self.m = int(len(self.pi))
        if num is None: #<A> is given
            num = self.m
            self.psi = num*self.psi
        self.num = int(num)
        if self.num<=0: raise ValueError('expected "num" to be a large positive number')
        self.gcc_in = self.get_init() # probability that row-group is on the in-component of the column-group
        self._sf_recursion, self._sf_recursion_apx = None, None
        recursion_lim = min(recursion_lim, self.num-1)
        self._sf_recursion, self._sf_recursion_apx = self.sf_recursion(recursion_lim), self.sf_recursion_apx(recursion_lim)
        self.dia = self.get_length('diameter')
        self._sf_recursion, self._sf_recursion_apx = self._sf_recursion[:self.get_lmax()+1], self._sf_recursion_apx[:self.get_lmax()+1]
        self.gcc_out = 1-self._sf_recursion[-1] # probability that column-group is on the (out-)component of the row-group
        self.gcc_out[np.isnan(self.gcc_out)] = 0

    def check_psipi(self, psi, pi, x=None, psi_args=()):
        if callable(psi):
            #discretize a continuous node space
            import scipy.integrate as integrate
            if not hasattr(pi, 'ppf'): raise ValueError('expected "pi" to be a random variable object with "cdf", "pdf" and "ppf" methods, such as scipy.stats.rv_continuous, that defines density in node space')
            warn('assuming kernel function induces sparsity i.e. psi will be normalized by number of nodes before drawing edges')
            if isinstance(x, int): x = [pi.ppf(i) for i in np.linspace(0, 1, x+1)]
            p = np.diff([pi.cdf(i) for i in x])
            if not (p>0).all(): raise ValueError('expected x to be an ordered partition of the node (sub)space')
            if psi_args: psi = [[integrate.dblquad(lambda a, b, *c: psi(a, b, *c)*pi.pdf(a)*pi.pdf(b), x[i], x[i+1], lambda a:x[j], lambda a:x[j+1], args=psi_args)[0]/(p[i]*p[j]) for j in range(len(x)-1)] for i in range(len(x)-1)]
            else: psi = [[integrate.dblquad(lambda a, b: psi(a, b)*pi.pdf(a)*pi.pdf(b), x[i], x[i+1], lambda a:x[j], lambda a:x[j+1])[0]/(p[i]*p[j]) for j in range(len(x)-1)] for i in range(len(x)-1)]
            pi = p/p.sum()
        if not isiterable(psi): psi = [[psi]]
        psi = np.array(psi, dtype=float)
        if pi is None: pi = [1/len(psi)]*len(psi)
        if not isiterable(pi): pi = [pi]
        pi = np.array(pi, dtype=float)
        if (psi<0).any(): raise ValueError('"psi" must have non-negative entries')
        if (pi<0).any(): raise ValueError('"pi" must have non-negative entries')
        if not np.allclose(sum(pi), 1): raise ValueError('entries of "pi" must sum up to 1')
        if len(psi.shape)!=2 or psi.shape[0]!=psi.shape[1]: raise ValueError('"psi" must be a square matrix')
        if len(pi.shape)!=1 or pi.shape[0]!=psi.shape[0]: raise ValueError('"pi" must be a vector of same length as "psi"')
        return psi, pi, x

    def get_length(self, kind='diameter', mean=False, low=1):
        pmf = self._pmf(np.arange(self.num), mean_source=mean, mean_target=mean, gcc_source=True, gcc_target=False, apx=False, init_corrected=True, init_scc=False, regular=False, w=1.)
        if kind=='mode': return np.argmax(pmf, axis=0) #set cutoff where the mode is achieved
        elif kind=='asymptotic': return self.num-1-np.argmax(np.logical_and(np.logical_not(np.isclose(pmf, 0)), pmf>=pmf[low])[::-1], axis=0) #set the cutoff just before when likelihood drops below P(\lambda=low)
        else: return self.num-1-np.argmax(np.logical_not(np.isclose(pmf, 0))[::-1], axis=0) #conservative cutoff once pmf "too close" to 0

    def get_lmax(self, k=None):
        if k is None: lmax = self.dia.max()
        else:
            if not isiterable(k): k = [k]
            k = np.array(k)
            if (k<0).any(): raise ValueError('expected distribution support to be non-negative')
            try: lmax = int(np.ceil((k[~np.isinf(k)].max())))
            except: lmax = self.dia.max()
        return lmax

    def get_support(self, lmax=None, scale=1, init=0): return np.arange(init, scale*self.get_lmax(lmax)+1)

    def get_gcc(self, mean=False, gcc_out=False, tol=1e-20):
        if gcc_out:
            def foo(rho=0.): return 1 - np.exp(-(rho*self.pi)@self.psi) #consistency equation for size of out-component in case of asymmetric block matrix
        else:
            def foo(rho=0.): return 1 - np.exp(-self.psi@(rho*self.pi)) #consistency equation for size of (in-)component
        rho_old, rho_new = np.ones(self.m), np.ones(self.m) # will only be able to find *one* non-trivial solution (if it exists)
        while True:
            rho_new = foo(rho_old)
            if ((rho_new-rho_old)**2).mean()<tol: break
            rho_old = rho_new
        if mean: rho_new = self.get_mean(rho_new)
        return rho_new

    def get_mean(self, x, axis=None, gcc=False):
        if isiterable(x):
            if not isinstance(x, np.ndarray): x = np.array(x)
            xs = np.array(x.shape)
            if axis is None:
                if not (xs==self.m).all(): raise ValueError('expected all input dimensions to be of length %i; did you forget to specify "axis"?'%self.m)
                if isinstance(gcc, bool): gcc = [gcc]*len(xs)
                if len(xs)>1: x = np.array([self.get_mean(x[i], gcc=gcc[1:]) for i in range(xs[0])])
                if gcc[0]: return x@(self.pi*self.get_gcc())/self.get_gcc(mean=True)
                else: return x@self.pi
            else:
                if gcc: factor = self.pi*self.get_gcc()/self.get_gcc(mean=True)
                else: factor = self.pi
                return (x*factor[(np.newaxis,)*axis+(slice(None),)+(np.newaxis,)*(len(xs)-axis-1)]).sum(axis=axis)
        else: raise ValueError('expected an input with at least one dimension of size %i'%self.m)

    def get_init(self):
        # get the initial condition for source node
        m = self.m
        pi = self.pi
        psi = self.psi.T # transpose to facilitate in-component for directed graphs
        num = self.num
        psi[psi>num] = num
        p_1 = psi/num
        sf_2, sf_1, omega_1 = np.ones((m, m)), 1-p_1, np.log2(1-p_1)
        for i in range(num-1):
            term = 1-(sf_2*(1-np.exp2(omega_1)))[:,:,np.newaxis]*psi[np.newaxis,:,:]/num
            term[term<0] = 0
            term[np.isnan(term)] = 0
            omega_curr = num*(np.log2(term)*pi[np.newaxis,:,np.newaxis]).sum(1)
            sf_curr = np.exp2(np.log2(sf_1)+omega_curr)
            sf_2 = sf_1.copy()
            sf_1 = sf_curr.copy()
            omega_1 = omega_curr.copy()
        out = 1-sf_curr
        out[np.isnan(out)] = 0
        return out.T

    def sf_recursion(self, k, apx=False, init_corrected=True, init_scc=False, regular=False, w=None):
        if apx: return self.sf_recursion_apx(k, init_corrected=init_corrected, init_scc=init_scc, regular=regular, w=w)
        if init_corrected and (not init_scc) and (not regular) and w is None and self._sf_recursion is not None: return self._sf_recursion
        m = self.m
        pi = self.pi
        psi = self.psi.copy()
        num = self.num
        psi[psi>num] = num
        if w is None: w = 1.
        if int(k)!=k or k<0: return
        sf, omega = [np.ones((m, m))], [np.zeros((m, m))]
        p_1 = psi/num
        if regular:
            psi = psi.copy()-1 # random regular graph "correction"
        elif init_corrected:
            gcc = self.gcc_in
            if init_scc: factor = 1-np.diag(gcc)[np.newaxis,:]
            else: factor = (1-(1-gcc)*(1-np.diag(gcc)[np.newaxis,:]))/gcc
            p_1 = p_1*factor
            p_1[np.isnan(p_1)]= 0
        if k>=1: 
            sf.append(1-p_1)
            omega.append(np.log2(1-p_1))
        for i in range(k-1):
            term = 1-(sf[-2]*(1-np.exp2(omega[-1])))[:,:,np.newaxis]*psi[np.newaxis,:,:]/num
            term[term<0] = 0
            term[np.isnan(term)] = 0
            omega_curr = num*(np.log2(term)*(pi*w)[np.newaxis,:,np.newaxis]).sum(1)
            sf_curr = np.exp2(np.log2(sf[-1])+omega_curr)
            sf.append(sf_curr)
            omega.append(omega_curr)
        return tuple(sf)

    def sf_recursion_apx(self, k, init_corrected=True, init_scc=False, regular=False, w=None):
        if init_corrected and (not init_scc) and (not regular) and w is None and self._sf_recursion_apx is not None: return self._sf_recursion_apx
        m = self.m
        pi = self.pi
        psi = self.psi
        num = self.num
        psi[psi>num] = num
        if w is None: w = 1.
        if int(k)!=k or k<0: return
        sf, omega = [np.ones((m, m))], [np.ones((m, m))]
        p_1 = psi/num
        if regular:
            psi = psi.copy()-1 # random regular graph "correction"
        elif init_corrected:
            gcc = self.gcc_in
            if init_scc: factor = 1-np.diag(gcc)[np.newaxis,:]
            else: factor = (1-(1-gcc)*(1-np.diag(gcc)[np.newaxis,:]))/gcc
            p_1 = p_1*factor
            p_1[np.isnan(p_1)]= 0
        if k>=1:
            sf.append(1-p_1)
            omega.append(p_1)
        for i in range(k-1):
            omega_curr =  omega[-1]@np.diag(pi*w)@psi
            sf_curr = sf[-1]*np.exp(-omega_curr)
            sf.append(sf_curr)
            omega.append(omega_curr)
        return tuple(sf)

    def parse_args(self, full=False, **kwargs):
        default = dict(mean_source=True, mean_target=True, gcc_source=True, gcc_target=False, apx=False, init_corrected=True, init_scc=False, regular=False, w=None)
        for key in default:
            if key in kwargs:
                default[key] = kwargs.pop(key)
        if full: return default.values(), kwargs
        else: return default.values()

    #def _argcheck(self, *args): return True

    def _sf(self, k, mean_source, mean_target, gcc_source, gcc_target, apx, init_corrected, init_scc, regular, w):
        lmax = self.get_lmax(k)
        out = self.sf_recursion(lmax, apx, init_corrected, init_scc, regular, w)
        if k is None: k = self.get_support()
        if not isiterable(k): k = [k]
        out = np.array([out[min(len(out)-1, int(i))] if i!=float('inf') else np.zeros((self.m, self.m)) for i in k])
        if not apx:
            if gcc_target: out = (out-1)/self.gcc_out + 1
            if not gcc_source: out =(out-1)*self.gcc_in + 1
            #if not gcc_source: out =(out-1)*(self.get_gcc()[:,np.newaxis]) + 1
        if mean_target: out = self.get_mean(out, axis=2, gcc=gcc_target and not apx)
        if mean_source: out = self.get_mean(out, axis=1, gcc=gcc_source and not apx)
        out = np.clip(out, 0, 1) #precision errors
        if len(k)==1: out = out[0]
        return out

    def _cdf(self, k, *args): return 1 - self._sf(k, *args)

    def _ppf(self, q, *args):
        out = self._cdf(self.get_support(), *args)
        out = np.concatenate([out, np.ones((1,)+out.shape[1:])], axis=0)
        if not isiterable(q): q = [q]
        out = np.stack([np.argmax(i<=out, axis=0) for i in q])
        idx = out>self.get_lmax()
        if idx.any():
            out = out.astype(float)
            out[idx] = np.inf
        if len(q)==1: out = out[0]
        return out

    def _pmf(self, k, mean_source, mean_target, gcc_source, gcc_target, apx, init_corrected, init_scc, regular, w):
        lmax = self.get_lmax(k)
        out = self.sf_recursion(lmax, apx, init_corrected, init_scc, regular, w)
        out = (1,) + out #to account for k=0
        if k is None: k = self.get_support()
        if not isiterable(k): k = [k]
        out = np.array([out[int(i)] - out[int(i)+1] if i!=float('inf') and i==int(i) and i<len(out)-1 else out[-1]*(i==float('inf')) for i in k])
        if not apx:
            if gcc_target: out = out/self.gcc_out
            if not gcc_source: out = out*self.gcc_in
            #if not gcc_source: out = out*(self.get_gcc()[:,np.newaxis])
        if mean_target: out = self.get_mean(out, axis=2, gcc=gcc_target and not apx)
        if mean_source: out = self.get_mean(out, axis=1, gcc=gcc_source and not apx)
        out[out<0] = 0. #precision-errors
        if len(k)==1: out = out[0]
        return out

    def _rvs(self, *args, size=None, random_state=None):
        if random_state is None: random_state = self._random_state
        U = random_state.uniform(size=size)
        Y = self._ppf(U, *args)
        return Y

    def sf(self, k=None, **kwds): return self._sf(k, *self.parse_args(**kwds))

    def cdf(self, k=None, **kwds): return self._cdf(k, *self.parse_args(**kwds))

    def pmf(self, k=None, **kwds): return self._pmf(k, *self.parse_args(**kwds))

    def ppf(self, k=None, **kwds): return self._ppf(k, *self.parse_args(**kwds))

    def rvs(self, **kwargs):
        args, kwargs = self.parse_args(True, **kwargs)
        return self._rvs(*args, **kwargs)

    def sample(self, num=10, path=True, bridge=False, all_components=False, regular=False, directed=False, tol=1e-1):
        from networkx import connected_components, neighbors, shortest_path_length, strongly_connected_components, weakly_connected_components
        n = self.num
        m = self.m
        if all_components or directed:
            if bridge: raise NotImplementedError('unable to generate empirical bridging distribution for all components or directed networks')
        if all_components and directed: raise NotImplementedError('unable to generate emprical shortest path length distribution for all components on directed networks')
        if regular:
            if m!=1: raise ValueError('can generate a random regular graph for m=1 only')
            if directed: raise NotImplementedError('unable to generate directed random regular directed graphs')
            from networkx.generators.random_graphs import random_regular_graph
        else:
            from .egosbm import EgocentricSBM
            mod = EgocentricSBM.StochasticBlockModel(self.psi, self.pi)        
        def get_G(directed=False):
            if regular: # sample a random regular graph
                G = random_regular_graph(d=int(self.psi[0,0]), n=n)
                for i in range(n): G.nodes[i]['block'] = 0 # dummy label to blockify over
                return G
            else:
                return mod.generate_networkdata(n, networkx_label='block', directed=directed)
        def get_gcc(G):
            from networkx.classes.digraph import DiGraph
            if isinstance(G, DiGraph):
                return max(weakly_connected_components(G), key=len)
                '''
                gcc_weak = set(max(weakly_connected_components(G), key=len)) # nodes on the weakly connected component (WCC)
                gcc_strong = sorted(strongly_connected_components(G), key=len, reverse=True) # nodes on the strongly connected component (SCC)
                gcc_strong_keep = []
                for i in range(min(len(gcc_strong), self.m)): # keep multiple SCCs in case of reducible model
                    if gcc_weak & set(gcc_strong[i]):
                        gcc_strong_keep += gcc_strong[i]
                    else:
                        if i>1: warn('including %i disconnected strongly connected components that map to the weakly connected component'%i)
                        break
                    if i==len(gcc_strong)-1: warn('including %i disconnected strongly connected components'%i)
                return {'gcc_strong': gcc_strong_keep, 'gcc_weak': list(gcc_weak)}'''
            else: return max(connected_components(G), key=len)
        def compute_deg(G, label='block'):
            blk = np.array([G.nodes[i][label] for i in range(n)])
            num = np.array([(blk==i).sum() for i in range(m)])
            gcc = get_gcc(G)
            blk_gcc = np.array([blk[i] for i in gcc])
            num_gcc = np.array([(blk_gcc==i).sum() for i in range(m)])
            deg = np.zeros((2, m, m))
            for i in range(n):
                for j in neighbors(G, i): deg[int(i not in gcc)][blk[i]][blk[j]] += 1
            return {'num':num, 'gcc':num_gcc, 'degree':deg}
        def compute_pmf(G, label='block'):
            # compute PMF for the giant component
            blk = np.array([G.nodes[i][label] for i in range(n)])
            num = np.array([(blk==i).sum() for i in range(m)])
            gcc = get_gcc(G)
            blk_gcc = np.array([blk[i] for i in gcc])
            num_gcc = np.array([(blk_gcc==i).sum() for i in range(m)])
            spl = dict(shortest_path_length(G))
            pmf = [np.zeros((m, m))]
            lmax = 0
            for i in gcc:
                for j in gcc:
                    if i!=j:
                        p_len = spl[i][j]
                        lmax = max(lmax, p_len)
                        if p_len>=len(pmf): pmf += [np.zeros((m, m)) for k in range(p_len-len(pmf)+1)]
                        pmf[p_len][blk[i]][blk[j]] += 1
            pmf = np.array(pmf)
            return {'num':num, 'gcc':num_gcc, 'path':pmf}
        def compute_pmf_directed(G, label='block'):
            blk = np.array([G.nodes[i][label] for i in range(n)])
            num = np.array([(blk==i).sum() for i in range(m)])
            gcc = np.array(list(get_gcc(G)))
            #blk_gcc = {key: np.array([blk[i] for i in gcc[key]]) for key in gcc}
            #num_gcc = {key: np.array([(blk_gcc[key]==i).sum() for i in range(m)]) for key in gcc}
            blk_gcc = np.array([blk[i] for i in gcc])
            idx = np.argsort(blk_gcc)
            gcc = gcc[idx]
            blk_gcc = blk_gcc[idx]
            num_gcc = {'gcc_weak': np.array([(blk_gcc==i).sum() for i in range(m)])}
            #for key in ['gcc_in', 'gcc_out']: num_gcc[key] = np.zeros(m) # nodes on the giant in-/out-components
            for key in ['gcc_in', 'gcc_out']: num_gcc[key] = np.zeros((m, m))
            track = {key: {i: np.zeros(m) for i in gcc} for key in ['gcc_in', 'gcc_out']}
            spl = dict(shortest_path_length(G))
            #first, infer which nodes are on the in-/out-components
            for i in gcc:
                for j in gcc:
                    if i!=j:
                        try:
                            p_len = spl[i][j]
                            track['gcc_in'][i][blk[j]] += 1
                            track['gcc_out'][j][blk[i]] += 1
                        except KeyError:
                            pass
            track_blk = {'gcc_in': [[] for i in range(m)], 'gcc_out': [[] for i in range(m)]}
            for key in ['gcc_in', 'gcc_out']:
                for node in track[key]:
                    track[key][node] = track[key][node]/num_gcc['gcc_weak']
                    track_blk[key][blk[node]].append(track[key][node])
                    track[key][node] = track[key][node]>tol # check if on the in-/out component
                for i in range(m):
                    track_blk[key][i] = np.array(track_blk[key][i])
            #next, compute the PMF for nodes on the in-/out-components
            pmf = [np.zeros((m, m))]
            lmax = 0
            for i in gcc:
                for key in ['gcc_in', 'gcc_out']: num_gcc[key][blk[i]] += track[key][i]
                for j in gcc:
                    if i!=j:
                        if track['gcc_in'][i][blk[j]] and track['gcc_out'][j][blk[i]]:
                            try:
                                p_len = spl[i][j]
                                lmax = max(lmax, p_len)
                                if p_len>=len(pmf): pmf += [np.zeros((m, m)) for k in range(p_len-len(pmf)+1)]
                                pmf[p_len][blk[i]][blk[j]] += 1
                            except KeyError:
                                pass
            num_gcc['gcc_out'] = num_gcc['gcc_out'].T # convenient so that target nodes are on the column
            # test 1
            '''
            # test 2
            gcc_diff = list(set(gccs['gcc_weak'])-set(gccs['gcc_strong']))
            for i in gccs['gcc_weak']:
                gcc_in, gcc_out = True, False
                for j in gccs['gcc_strong']:
                    if i!=j:
                        try:
                            p_len = spl[i][j]
                            lmax = max(lmax, p_len)
                            if p_len>=len(pmf): pmf += [np.zeros((m, m)) for k in range(p_len-len(pmf)+1)]
                            pmf[p_len][blk[i]][blk[j]] += 1
                        except KeyError: # node is not on in-component
                            try:
                                p_len = spl[j][i]
                                gcc_in = False
                                gcc_out = True # path from SCC to node exists, must be on the out-component
                                break # it cannot have a path to any other node on the SCC
                            except KeyError:
                                gcc_in = False # on neither the in- nor the out-components (like a "collider" structure)
                                break
                if gcc_in or gcc_out:
                    for j in gcc_diff:
                        if i!=j:
                            try:
                                p_len = spl[i][j]
                                lmax = max(lmax, p_len)
                                if p_len>=len(pmf): pmf += [np.zeros((m, m)) for k in range(p_len-len(pmf)+1)]
                                pmf[p_len][blk[i]][blk[j]] += 1
                            except KeyError:
                                pass
                    if i in gccs['gcc_strong']:
                        for key in ['gcc_in', 'gcc_out']: num_gcc[key][blk[i]] += 1
                    elif gcc_out:
                        #if blk[i]==0: print(blk[i], G.out_degree(i), G.in_degree(i), [(blk[j], G.out_degree(j), G.in_degree(j)) for j in neighbors(G, i)])
                        num_gcc['gcc_out'][blk[i]] += 1
                    else: num_gcc['gcc_in'][blk[i]] += 1'''
            pmf = np.array(pmf)
            return {'num':num, 'gcc':num_gcc, 'path':pmf, 'track':track_blk}
        def compute_pmf_all(G, label='block'):
            # compute PMF for all components
            blk = np.array([G.nodes[i][label] for i in range(n)])
            num = np.array([(blk==i).sum() for i in range(m)])
            gcc = get_gcc(G)
            blk_gcc = np.array([np.array([blk[i] for i in gcc_i]) for gcc_i in gcc])
            num_gcc = np.array([np.array([(blk_gcc_i==i).sum() for i in range(m)]) for blk_gcc_i in blk_gcc])
            spl = dict(shortest_path_length(G))
            pmf = [np.zeros((m, m))]
            lmax = 0
            for gcc_i in gcc:
                for i in gcc_i:
                    for j in gcc_i:
                        if i!=j:
                            p_len = spl[i][j]
                            lmax = max(lmax, p_len)
                            if p_len>=len(pmf): pmf += [np.zeros((m, m)) for k in range(p_len-len(pmf)+1)]
                            pmf[p_len][blk[i]][blk[j]] += 1
            pmf = np.array(pmf)
            return {'num':num, 'gcc':num_gcc, 'path':pmf}
        def compute_bridge(G, label='block'):        
            blk = np.array([G.nodes[i][label] for i in range(n)])
            num = np.array([(blk==i).sum() for i in range(m)])
            gcc = get_gcc(G)
            blk_gcc = np.array([blk[i] for i in gcc])
            num_gcc = np.array([(blk_gcc==i).sum() for i in range(m)])
            spl = dict(shortest_path_length(G))
            pmf = [np.zeros((m, m))]
            via = [np.zeros((1, m))]
            unq = [np.zeros((1, 2, m))]
            lmax = 0
            for i in gcc:
                for j in gcc:
                    if i!=j:
                        p_len = spl[i][j]
                        lmax = max(lmax, p_len)
                        if p_len>=len(pmf):
                            pmf += [np.zeros((m, m)) for k in range(p_len-len(pmf)+1)]
                            if p_len>len(via):
                                via += [np.zeros((k+1, m)) for k in range(len(via), p_len)]
                                unq += [np.zeros((k+1, 2, m)) for k in range(len(unq), p_len)]
                        pmf[p_len][blk[i]][blk[j]] += 1
                        num_p = [[] for i in range(p_len)]
                        for k in gcc:
                            if k!=i and k!=j:
                                if spl[i][k]+spl[k][j]==p_len:
                                    via[p_len-1][spl[i][k]][blk[k]] += 1
                                    num_p[spl[i][k]].append(blk[k])
                                else: via[p_len-1][0][blk[k]] += 1
                        for u in range(1, len(num_p)):
                            if len(num_p[u])==1: unq[p_len-1][u][1][num_p[u][0]] += 1
                            else:
                                for v in num_p[u]: unq[p_len-1][u][0][v] += 1
            return {'num':num, 'gcc':num_gcc, 'path':pmf, 'bridge':via, 'unique':unq}
        def sample_deg():
            out = {'num':[], 'gcc':[], 'degree':[]}
            for i in range(num):
                G = get_G()
                y = compute_deg(G)
                for key in out: out[key].append(y[key])
            for key in out: out[key] = np.stack(out[key], axis=0)
            return out
        def sample_pmf(all_components=False, directed=False):
            out = {'num':[], 'gcc':[], 'path':[], 'track':[]} # remove 'track'
            lmax = 0
            for i in range(num):
                G = get_G(directed)
                if all_components: y = compute_pmf_all(G)
                elif directed: y = compute_pmf_directed(G)
                else: y = compute_pmf(G)
                for key in out: out[key].append(y[key])
                lmax = max(len(y['path']), lmax)
            for i in range(num):
                if len(out['path'][i])<lmax: out['path'][i] = np.concatenate([out['path'][i], np.nan*np.ones((lmax-len(out['path'][i]), m, m))], axis=0)
            for key in out:
                if key=='gcc' and all_components: continue
                elif key=='gcc' and directed: out[key] = {k: np.stack([o[k] for o in out[key]], axis=0) for k in out[key][0]}
                elif key=='track' and directed: continue
                else: out[key] = np.stack(out[key], axis=0)
            return out
        def sample_bridge():
            out = {'num':[], 'gcc':[], 'path':[], 'bridge':[], 'unique':[]}
            lmax = 0
            for i in range(num):
                G = get_G()
                y = compute_bridge(G)
                for key in out: out[key].append(y[key])
                lmax = max(len(y['path']), lmax)
            for i in range(num):
                if len(out['path'][i])<lmax: out['path'][i] = np.concatenate([out['path'][i], np.nan*np.ones((lmax-len(out['path'][i]), m, m))], axis=0)
                for j in range(len(out['bridge'][i]), lmax):
                    out['bridge'][i].append(np.nan*np.ones((j+1, m)))
                    out['unique'][i].append(np.nan*np.ones((j+1, 2, m)))
            for key in out:
                if key in ['bridge', 'unique']: out[key] = [np.stack([out[key][k][i] for k in range(num)], axis=0) for i in range(lmax)]
                else: out[key] = np.stack(out[key], axis=0)
            return out
        if bridge: return sample_bridge()
        elif path: return sample_pmf(all_components, directed)
        else: return sample_deg()

    def add_sample(self, samples=None, num=1, **kwargs):
        def align_samples(x, y):
            if isinstance(x, list) and isinstance(y, list):
                l1, l2 = len(x), len(y)
                if l1>l2: y = y + [np.nan*np.ones((y[0].shape[0], i+1, self.m)) for i in range(l2, l1)]
                elif l2>l1: x = x + [np.nan*np.ones((x[0].shape[0], i+1, self.m)) for i in range(l1, l2)]
                return [np.concatenate([x_i, y_i], axis=0) for x_i, y_i in zip(x, y)]
            elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and len(x.shape)==len(y.shape):
                l1, l2 = x.shape[1], y.shape[1]
                if l1>l2: y = np.concatenate([y, np.nan*np.ones((len(y), l1-l2, self.m, self.m))], axis=1)
                elif l2>l1: x = np.concatenate([x, np.nan*np.ones((len(x), l2-l1, self.m, self.m))], axis=1)
                return np.concatenate([x, y], axis=0)
            else: raise ValueError('expected previous sample to be of the same kind as the new one; did you forget to set "bridge"?')
        if samples is None: return self.sample(num, **kwargs)
        else:
            new = self.sample(num, bridge='bridge' in samples, **kwargs)
            return {kind: align_samples(samples[kind], new[kind]) for kind in samples}

    def sum_sample(self, samples=10, cdf=False, bridge=False, mean_source=False, mean_target=False, mean_bridge=False, bridge_length=0, gcc_source=False, gcc_target=False, gcc_bridge=False, all_components=False, regular=False, directed=False, ignore_nan=False, **kwargs):
        if samples is None: return None
        if isinstance(samples, int): samples = self.sample(samples, bridge=bridge, all_components=all_components, regular=regular, directed=directed, **kwargs)        
        def foo_path():
            if isinstance(samples['gcc'], dict): return foo_path_directed()
            elif isinstance(samples['gcc'], list): return foo_path_all()
            out_comp_t = samples['gcc'][:,:,np.newaxis]*((samples['num']-samples['gcc'])[:,np.newaxis,:])
            out_comp_s = samples['gcc'][:,np.newaxis,:]*((samples['num']-samples['gcc'])[:,:,np.newaxis])
            out_comp_st = ((samples['num']-samples['gcc'])[:,np.newaxis,:])*((samples['num']-samples['gcc'])[:,:,np.newaxis])
            y = samples['path'].copy()
            y[:,0] = int(not gcc_target)*out_comp_t + int(not gcc_source)*out_comp_s + int(not gcc_source and not gcc_target)*out_comp_st
            if mean_target: y = y.sum(axis=3)
            if mean_source: y = y.sum(axis=2)
            y = y/(np.nansum(y, axis=1)[(slice(None), np.newaxis)+(slice(None),)*(y.ndim-2)])
            y[:,0] = 0
            return y
        def foo_path_directed():
            out_comp_t = samples['gcc']['gcc_in']*(samples['num'][:,:,np.newaxis]-samples['gcc']['gcc_out'])
            out_comp_s = samples['gcc']['gcc_out']*(samples['num'][:,:,np.newaxis]-samples['gcc']['gcc_in'])
            out_comp_st = (samples['num'][:,:,np.newaxis]-samples['gcc']['gcc_out'])*(samples['num'][:,:,np.newaxis]-samples['gcc']['gcc_in'])
            y = samples['path'].copy()
            y[:,0] = int(not gcc_target)*out_comp_t + int(not gcc_source)*out_comp_s + int(not gcc_source and not gcc_target)*out_comp_st
            if mean_target: y = y.sum(axis=3)
            if mean_source: y = y.sum(axis=2)
            y = y/(np.nansum(y, axis=1)[(slice(None), np.newaxis)+(slice(None),)*(y.ndim-2)])
            y[:,0] = 0
            return y
        def foo_path_all():
            num_samples = len(samples['gcc'])
            out_comp = np.stack([(samples['gcc'][i][:,:,np.newaxis]*((samples['num'][i]-samples['gcc'][i])[:,np.newaxis,:])).sum(axis=0) for i in range(num_samples)], axis=0)
            y = samples['path'].copy()
            y[:,0] = int(not gcc_target)*out_comp
            if mean_target: y = y.sum(axis=3)
            if mean_source: y = y.sum(axis=2)
            y = y/(np.nansum(y, axis=1)[(slice(None), np.newaxis)+(slice(None),)*(y.ndim-2)])
            y[:,0] = 0
            return y
        def foo_bridge():
            if gcc_source!=gcc_target: raise ValueError('expected "gcc_source" and "gcc_target" to be the same, but received conflicting values instead: %s, %s'%(gcc_source, gcc_target))
            num, num_gcc = samples['num'].sum(axis=1)[:,np.newaxis,np.newaxis], samples['gcc'].sum(axis=1)[:,np.newaxis,np.newaxis]
            out_comp_st = 2*(num_gcc-1)*(num-num_gcc)*samples['gcc'][:,np.newaxis,:]
            out_comp_b = (num_gcc-1)*num_gcc*(samples['num']-samples['gcc'])[:,np.newaxis,:]
            out_comp_stb = (num-2)*(num-num_gcc-1)*(samples['num']-samples['gcc'])[:,np.newaxis,:]
            y = [i.copy() for i in samples['bridge']]
            lmax = len(y)
            y[0] = (int(not gcc_source and not gcc_target))*out_comp_st + int(not gcc_bridge)*out_comp_b + int(not gcc_source and not gcc_target and not gcc_bridge)*out_comp_stb
            if mean_bridge: y = [i.sum(axis=2) for i in y]
            y = np.concatenate(y, axis=1)
            n = y.shape[0]
            y = y/(np.nansum(y, axis=1)[(slice(None), np.newaxis)+(slice(None),)*(y.ndim-2)])
            if not bridge_length:
                z = [np.zeros((n,)+y.shape[2:])]
                tmp = 0
                for i in range(1, lmax+1):
                    y[:,tmp] = 0
                    z.append(np.nansum(y[:,tmp:tmp+i], axis=1))
                    tmp += i
                z = np.stack(z, axis=1)
            else:
                if not 1<=bridge_length<=lmax or int(bridge_length)!=bridge_length: raise ValueError('expected "bridge_length" to be an int between 1 and %i'%lmax)
                tmp = bridge_length*(bridge_length-1)//2
                y[:,tmp] = 0
                z = np.concatenate([y[:,tmp:tmp+bridge_length], np.zeros((n, 1)+y.shape[2:])], axis=1)
            return z
        if bridge: samples = foo_bridge()
        else: samples = foo_path()
        if ignore_nan:
            if cdf: samples = np.nancumsum(samples, axis=1)
            mu, sigma = np.nanmean(samples, axis=0), np.nanstd(samples, axis=0)
        else:
            if cdf: samples = np.cumsum(samples, axis=1)
            mu, sigma = np.mean(samples, axis=0), np.std(samples, axis=0)
        return mu, sigma

    def plot_unique_bridge(self, samples, bridge_length=0, mean_bridge=False, ignore_nan=False, ax=None, legend=False, grid=True, dpi=None):
        import numpy as np
        from matplotlib.ticker import MaxNLocator
        y = samples['unique']
        lmax = len(y)
        if bridge_length:
            if not 1<=bridge_length<=lmax or int(bridge_length)!=bridge_length: raise ValueError('expected "bridge_length" to be an int between 1 and %i'%lmax)
            y = y[bridge_length-1]
        else: y = np.stack([i.sum(axis=1) for i in y], axis=1)
        if mean_bridge: y = np.nansum(y, axis=3)[:,:,:,np.newaxis]
        y = y[:,:,1]/np.nansum(y, axis=2)
        if ignore_nan: mu, sigma = np.nanmean(y, axis=0), np.nanstd(y, axis=0)
        else: mu, sigma = np.mean(y, axis=0), np.std(y, axis=0)
        x = np.arange(mu.shape[0])
        no_axis = ax is None
        if no_axis:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(dpi=dpi)
        for i in range(mu.shape[1]): ax.errorbar(x, mu[:,i], yerr=sigma[:,i], label=i)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(grid)
        if legend: ax.legend()
        if no_axis: return fig, ax

    def plot(self, lmax=None, samples=None, cdf=True, bridge=False, bridge_length=0, mean_source=False, mean_target=False, mean_bridge=False, gcc_source=False, gcc_target=False, gcc_bridge=False, all_components=False,
        gcc_size=False, apx=False, init_corrected=True, init_scc=False, regular=False, bridge_corrected=True, log=False, lengths=False, subset=[], source=True, grid=False, figsize=(4, 3), dpi=None, 
        marker=False, sampleevery=1, linewidth=1.5, ls='-', color=False, cmap='coolwarm', legend=False, legend_loc='upper left', label='', label_sample='', label_suffix='', title='', label_cbar='Block Index', label_x='$l$', label_y=True, cbar=True, ax=None):
        import numpy as np
        from matplotlib.ticker import MaxNLocator
        gcc_tot = source and mean_target
        if gcc_size: phi = self.get_gcc(mean=gcc_tot, gcc_out=True)
        if bridge:
            mean_source, mean_target = True, True
            gcc_source, gcc_target = gcc_bridge, gcc_bridge
            if bridge_length:
                cdf = False
                lmax = bridge_length
        samples = self.sum_sample(samples, cdf=cdf, bridge=bridge, mean_source=mean_source, mean_target=mean_target, mean_bridge=mean_bridge, bridge_length=bridge_length, gcc_source=gcc_source, gcc_target=gcc_target, gcc_bridge=gcc_bridge, all_components=all_components)
        x = self.get_support(lmax=lmax)
        if bridge:
            #if gcc_source!=gcc_target: warn('interpret with caution as source and target nodes are conditioned differently on the connected component as %s and %s respectively'%(gcc_source, gcc_target))
            y = self.pmf_bridge(x, mean_bridge=mean_bridge, mean_length=not bool(bridge_length), gcc_source=gcc_source, gcc_target=gcc_target, gcc_bridge=gcc_bridge, apx=apx, init_corrected=init_corrected, independent=not bridge_corrected)
            if bridge_length: y = y[bridge_length]
            if cdf: y = y.cumsum(axis=0)
            if mean_bridge:
                y = y[:,np.newaxis]
                labels = [0]
                if samples is not None: samples = (samples[0][:,np.newaxis], samples[1][:,np.newaxis])
            else: labels = list(range(self.m))
        else:
            if cdf: y = self.cdf(x, mean_source=mean_source, mean_target=mean_target, gcc_source=gcc_source, gcc_target=gcc_target, apx=apx, init_corrected=init_corrected, init_scc=init_scc, regular=regular)
            else: y = self.pmf(x, mean_source=mean_source, mean_target=mean_target, gcc_source=gcc_source, gcc_target=gcc_target, apx=apx, init_corrected=init_corrected, init_scc=init_scc, regular=regular)
            if not source:
                def flip(x): return np.array([i.T for i in x])
                y = flip(y)
                if samples is not None: samples = (flip(samples[0]), flip(samples[1]))
            if not mean_source and not mean_target:
                from itertools import product
                labels = list(product(*[range(self.m)]*2))
            else:
                if mean_source and mean_target:
                    y = y[:,np.newaxis]
                    labels = [0]
                    samples = (samples[0][:,np.newaxis], samples[1][:,np.newaxis])
                else: labels = list(range(self.m))
        if isinstance(subset, (int, tuple)): subset = [subset]
        if subset:
            if not bridge and not mean_source and not mean_target:
                from itertools import chain
                subset = list(chain(*[list(zip([i]*self.m, range(self.m))) if isinstance(i, int) else [tuple(i)] for i in subset]))
        no_axis = ax is None
        if no_axis:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        if color:
            if subset: vals = range(len(subset))
            else: vals = range(len(labels))
            if isinstance(color, str): colors = [color]*len(vals)
            elif isiterable(color): colors = color
            elif cbar: colors = get_color(vals, ax=ax, ticks=MaxNLocator(integer=True), label=label_cbar, cmap=cmap)
            else: colors = get_color(vals, ax=None, ticks=MaxNLocator(integer=True), label=label_cbar, cmap=cmap)
        if marker:
            if subset: l = len(subset)
            else: l = len(labels)
            if isinstance(marker, str): marker = [marker]*l
            elif isinstance(marker, bool): marker = get_marker(l)
        it = 0
        for i in labels:
            if label: lab = label
            else: lab = '%s%s'%(i, label_suffix)
            if subset and i not in subset: continue
            if color: ax.plot(x, [y[j][i] for j in range(len(y))], label=lab, c=colors[it], ls=ls, lw=linewidth)
            else: ax.plot(x, [y[j][i] for j in range(len(y))], label=lab, ls=ls, lw=linewidth)
            if samples is not None:
                c = ax.lines[-1].get_color()
                support = range(0, min(x[-1]+1, len(samples[0])), sampleevery)
                if marker: ax.errorbar(support, [samples[0][j][i] for j in support], yerr=[samples[1][j][i] for j in support], c=c, ls='', marker=marker[it], label=label_sample)
                else: ax.plot(support, [samples[0][j][i] for j in support], c=c, ls='--', lw=linewidth, label=label_sample)
            if gcc_size and not gcc_tot:
                c = ax.lines[-1].get_color()
                if isinstance(i, int): ax.axhline(phi[i], c=c, ls='--', lw=linewidth)
                elif source: ax.axhline(phi[i[1]], c=c, ls='--', lw=linewidth)
                else: ax.axhline(phi[i[0]], c='k', ls='--', lw=linewidth)
            it += 1
        if gcc_size and gcc_tot: ax.axhline(phi, c='k', ls='--', lw=linewidth)
        if lengths:
            ax.axvline(self.get_length('mode', mean=True), color='k', ls='--', lw=linewidth)
            ax.axvline(self.get_length('asymptotic', mean=True), color='k', ls='-.', lw=linewidth)
            ax.axvline(self.get_length('diameter', mean=True), color='k', ls=':', lw=linewidth)
        if label_x: ax.set_xlabel(str(label_x))
        if label_y:
            if isinstance(label_y, bool):
                if bridge:
                    if bridge_length: sym = 'beta_k^{%i}'%bridge_length
                    else: sym = 'beta_k'
                else: sym = 'lambda_{ij}'
                if cdf: opn = '<='
                else: opn = '='
                label_y = '$P(\\%s%sl)$'%(sym, opn)
            ax.set_ylabel(str(label_y))
        if title: ax.set_title(str(title))
        if log: ax.set_yscale('log')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(grid)
        if legend:
            if legend_loc: ax.legend(loc=legend_loc)
            else: ax.legend()
        if no_axis: return fig, ax

    def plot_grid(self, samples=None, lmax=None, subset=[], gcc_source=True, gcc_target=False, init_corrected=True, init_scc=False, all_components=False, mean=False, legend=False, color=False, marker=False, sampleevery=1, linewidth=1.5, figsize=(2, 3), dpi=120, sharey=False, legend_anchor=(1.65, 0.5), label=True, title=True, save='', apx=True, ax=None, **kwargs):
        if isinstance(subset, int): subset = [subset]
        if not subset: subset = list(range(self.m))
        if title:
            if isinstance(title, bool): title = ['$P(\\lambda_{ij}=l)$', '$P(\\lambda_{ij}\\le l)$']
            elif isinstance(title, str): title = [title]*2
        else: title = ['', '']
        no_axis = ax is None
        if no_axis:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(nrows=len(subset), ncols=2, squeeze=False, sharex=True, sharey=sharey, figsize=(2*figsize[1], len(subset)*figsize[0]), dpi=dpi, constrained_layout=True)
        for i in range(len(subset)):
            if label: lab = ['Block %i'%(subset[i]+1), '']
            else: lab  = ['', '']
            for j in [0, 1]:
                self.plot(samples=samples, lmax=lmax, subset=subset[i], gcc_size=bool(j) and not(all_components), gcc_source=gcc_source, gcc_target=gcc_target, cdf=bool(j), apx=False, init_corrected=init_corrected, init_scc=init_scc, all_components=all_components, mean_target=mean, color=color, marker=marker, sampleevery=sampleevery, linewidth=linewidth, label_x=False, label_y=lab[j], cbar=False, legend=False, ax=ax[i,j], **kwargs)
                if apx: self.plot(subset=subset[i], lmax=lmax, gcc_size=bool(j) and not(all_components), gcc_source=gcc_source, gcc_target=gcc_target, cdf=bool(j), apx=True, init_corrected=False, all_components=all_components, mean_target=mean, color=color, marker=marker, linewidth=linewidth, ls='-.', label_x=False, label_y=False, cbar=False, legend=False, ax=ax[i,j], **kwargs)
                if i==0: ax[i,j].set_title(title[j])
                if label and i==len(subset)-1: ax[i,j].set_xlabel('$l$')
        if no_axis and legend and not mean:
            ax_outer = fig.add_subplot(111, frame_on=False, xticks=[], yticks=[])
            handles, labels = ax[0,0].get_legend_handles_labels()
            ax_outer.legend(handles=handles, labels=range(len(labels)), bbox_to_anchor=legend_anchor, loc='center')
        if save: plt.savefig(save, bbox_inches='tight', dpi=dpi)
        if no_axis: return fig, ax

    def pmf_bridge(self, x=None, mean_source=True, mean_target=True, mean_bridge=False, mean_length=True, gcc_source=False, gcc_target=False, gcc_bridge=False, apx=False, init_corrected=True, independent=False):
        if x is None: x = self.get_support()
        if not isiterable(x): x = [x]
        tmp = self.get_lmax(x)
        dia = self.get_lmax()
        #w = self.sf(mean_source=False, mean_target=False, gcc_source=True, gcc_target=False, apx=apx, init_corrected=init_corrected)
        log_sf = np.log2(self.sf(mean_source=False, mean_target=False, gcc_source=True, gcc_target=True, apx=apx, init_corrected=init_corrected))
        #log_sf = np.log2(self.sf(mean_source=False, mean_target=False, gcc_source=True, gcc_target=False, apx=apx, init_corrected=init_corrected))
        log_pmf = np.log2(self.pmf(mean_source=False, mean_target=False, gcc_source=True, gcc_target=True, apx=apx, init_corrected=init_corrected))
        log_pmf_nongcc = np.log2(self.pmf(mean_source=False, mean_target=False, gcc_source=False, gcc_target=True, apx=apx, init_corrected=init_corrected))
        #log_pmf = np.log2(self.pmf(mean_source=False, mean_target=False, gcc_source=True, gcc_target=False, apx=apx, init_corrected=init_corrected))
        pmf = self.pmf(mean_source=False, mean_target=False, gcc_source=True, gcc_target=False, apx=apx, init_corrected=init_corrected)
        #pmf = self.pmf(mean_source=False, mean_target=False, gcc_source=False, gcc_target=False, apx=apx, init_corrected=init_corrected)
        sf = self.sf(mean_source=False, mean_target=False, gcc_source=True, gcc_target=False, apx=apx, init_corrected=init_corrected)
        sf_gcc = self.sf(mean_source=False, mean_target=False, gcc_source=True, gcc_target=True, apx=apx, init_corrected=init_corrected)
        log_sf_gcc = np.log2(sf_gcc)
        cdf_gcc = 1-sf_gcc
        log_cdf_gcc = np.log2(cdf_gcc)
        pmf_gcc = np.diff(cdf_gcc, axis=0, prepend=0)
        log_pmf_gcc = np.log2(pmf_gcc)
        #if apx: pmf = -np.diff(np.log2(sf), axis=0, prepend=0)
        #else: pmf = -np.diff(sf, axis=0, prepend=1)
        psi = self.psi
        log_psi = np.log2(psi)
        pi = self.pi
        gcc = self.get_gcc()
        psi_gcc = psi*(1+(1/gcc[np.newaxis,:]-1)*gcc[:,np.newaxis])
        log_gcc = np.log2(self.gcc_out)
        gcc_inv = 1-self.gcc_out
        log_n = np.log2(self.num)
        if False:#not independent:
            nbr = [[np.log2(self.pmf(mean_source=False, mean_target=False, gcc_source=True, gcc_target=True, apx=apx, init_corrected=init_corrected, w=w[i,j,:])) for j in range(self.m)] for i in range(1, dia)]
            nbr_to = [np.stack(i, axis=-1) for i in nbr]
            nbr_fr = [np.stack(i, axis=1) for i in nbr]
        def brg3(a, b):
            if a==1 or independent: out = log_pmf[a][:,:,np.newaxis]
            else:
                if apx: out = np.log2(np.stack([sf[a-1]*(1-np.exp2(-pmf[a-1]@np.diag(pi*sf[b][i,:])@psi)) for i in range(self.m)], axis=-1))
                #else: out = np.log2(np.stack([sf[a-1]*(1-np.exp2(self.num*(np.log2(1-pmf[a-1][:,:,np.newaxis]*psi[np.newaxis,:,:]*sf[b][i,:][np.newaxis,:,np.newaxis]/self.num)*pi[np.newaxis,:,np.newaxis]).sum(1)))/(gcc[np.newaxis,:]) for i in range(self.m)], axis=-1))
                else:
                    #t1 = [1-np.exp2(self.num*(np.log2(1-pmf[a-1][:,:,np.newaxis]*psi[np.newaxis,:,:]*sf_gcc[b][i,:][np.newaxis,:,np.newaxis]/self.num)*(pi)[np.newaxis,:,np.newaxis]).sum(1)) for i in range(self.m)]
                    t1 = [1-np.exp2(self.num*(np.log2(1-pmf[a-1][:,:,np.newaxis]*psi[np.newaxis,:,:]*sf_gcc[b][:,i][np.newaxis,:,np.newaxis]/self.num)*(pi)[np.newaxis,:,np.newaxis]).sum(1)) for i in range(self.m)]
                    #t2 = [np.exp2(self.num*(np.log2(1-pmf[a-1][:,:,np.newaxis]*psi[np.newaxis,:,:]*(1-sf_gcc[b][i,:][np.newaxis,:,np.newaxis])/self.num)*(pi)[np.newaxis,:,np.newaxis]).sum(1)) for i in range(self.m)]
                    #out = np.log2(np.stack([sf_gcc[a-1]*t1[i] for i in range(self.m)], axis=-1))
                    out = np.log2(np.stack([sf[a-1]*t1[i]/self.gcc_out for i in range(self.m)], axis=-1))
                    #tmp = - np.log2(1-gcc_inv/sf[a-1])[:,:,np.newaxis]
                    #out = out + tmp
            return out+log_pmf[b][np.newaxis,:,:]+log_sf[a+b-1][:,np.newaxis,:]
        def brg4(a, b):
            if a==1 or independent: out = log_pmf[a][:,:,np.newaxis]
            else:
                if apx: out = np.log2(np.stack([1-np.exp2(-pmf[a-1]@np.diag(pi*sf[a+b-2][i,:])@psi) for i in range(self.m)], axis=-1))
                #else: out = np.log2(np.stack([sf[a-1]*(1-np.exp2(self.num*(np.log2(1-pmf[a-1][:,:,np.newaxis]*psi[np.newaxis,:,:]*sf[b][i,:][np.newaxis,:,np.newaxis]/self.num)*pi[np.newaxis,:,np.newaxis]).sum(1)))/(gcc[np.newaxis,:]) for i in range(self.m)], axis=-1))
                else:
                    #out = np.log2(np.stack([1-np.exp2(self.num*(np.log2(1-pmf[a-1][:,:,np.newaxis]*psi[np.newaxis,:,:]*sf[a+b-2][i,:][np.newaxis,:,np.newaxis]/self.num)*pi[np.newaxis,:,np.newaxis]).sum(1)) for i in range(self.m)], axis=-1))
                    factor = [1 for i in range(self.m)]#[sf[a-1][:,np.newaxis,:]*sf[a+b-2][i,:][np.newaxis,:,np.newaxis] for i in range(self.m)]
                    out = np.log2(np.stack([1-np.exp2(self.num*(np.log2(1-pmf[a-1][:,:,np.newaxis]*psi[np.newaxis,:,:]*factor[i]/self.num)*pi[np.newaxis,:,np.newaxis]).sum(1)) for i in range(self.m)], axis=-1))
                    #out = np.log2(np.stack([1-np.exp2(self.num*(np.log2(1-(pmf[a-1][:,:,np.newaxis]*psi[np.newaxis,:,:]*sf[a+b-2][i,:][np.newaxis,:,np.newaxis])/((1-gcc_inv/sf[a-1])[:,np.newaxis,:]*self.num))*pi[np.newaxis,:,np.newaxis]).sum(1)) for i in range(self.m)], axis=-1))
                    tmp = - np.log2(1-gcc_inv/sf[a-1])[:,:,np.newaxis]
                    #print(a, tmp)
                    out = out + tmp
                    #if gcc_bridge: out - gcc_log[np.newaxis,:,np.newaxis]
            return out+log_pmf[b][np.newaxis,:,:]+log_sf[a+b-1][:,np.newaxis,:]
        def brg2(a, b):
            if b==1 or independent: out = log_pmf[b][np.newaxis,:,:]
            else:
                if apx: out = np.log2(np.stack([1-np.exp2(-pmf[b-1]@np.diag(pi*sf[a+b-2][i,:])@psi) for i in range(self.m)], axis=0))
                #else: return np.log2(np.stack([sf[a-1]*(1-np.exp2(self.num*(np.log2(1-pmf[a-1][:,:,np.newaxis]*psi[np.newaxis,:,:]*sf[b][i,:][np.newaxis,:,np.newaxis]/self.num)*pi[np.newaxis,:,np.newaxis]).sum(1)))/(gcc[np.newaxis,:]) for i in range(self.m)], axis=-1))
                else: out = np.log2(np.stack([1-np.exp2(self.num*(np.log2(1-pmf[b-1][:,:,np.newaxis]*psi[np.newaxis,:,:]*sf[a+b-2][i,:][np.newaxis,:,np.newaxis]/self.num)*pi[np.newaxis,:,np.newaxis]).sum(1)) for i in range(self.m)], axis=0))
            if gcc_target: out = out/gcc[np.newaxis,np.newaxis,:]
            return out+log_pmf[a][:,:,np.newaxis]+log_sf[a+b-1][:,np.newaxis,:]
        def brg(a, b):
            if b==1 or independent: out = log_pmf[b][np.newaxis,:,:]
            else:
                idx = int(np.abs(b-a-1))
                if idx>0: f_iu = np.log2(1+np.exp2(log_cdf_gcc[idx-1]-log_sf_gcc[a+b-2]))
                else: f_iu = 0
                f_iu = f_iu+log_sf[a+b-2]
                g_ikj = (log_sf[b-1]-log_gcc)[np.newaxis,:,:]+(np.log2(1-pmf_gcc[a+b-2]-pmf_gcc[a+b-1])-log_sf_gcc[a+b-1])[:,np.newaxis,:]
                tmp = [1-np.exp2(log_pmf[b-1][:,:,np.newaxis]+log_psi[np.newaxis,:,:]+f_iu[i][np.newaxis,:,np.newaxis]+g_ikj[i][:,np.newaxis,:]-log_n) for i in range(self.m)]
                for i in range(self.m): tmp[i][tmp[i]<0] = 0. #capping for very long lengths
                out = np.log2(np.stack([1-np.exp2(self.num*(np.log2(tmp[i])*pi[np.newaxis,:,np.newaxis]).sum(1)) for i in range(self.m)], axis=0)) #marginalize over u
            return out+log_pmf[a][:,:,np.newaxis]+log_sf[a+b-1][:,np.newaxis,:]
        def brg_(a, b):
            if b==1 or independent: out = log_pmf[b][np.newaxis,:,:]
            else:
                idx = int(np.abs(b-a-1))
                if idx>0: f_iu = 1+np.exp2(np.log2(cdf_gcc[idx-1])-np.log2(sf_gcc[a+b-2]))
                else: f_iu = 1
                f_iu = f_iu*sf[a+b-2]
                g_ikj = sf[b-1][np.newaxis,:,:]*((1-pmf_gcc[a+b-2]-pmf_gcc[a+b-1])/sf[a+b-1])[:,np.newaxis,:]
                out = np.log2(np.stack([1-np.exp2(self.num*(np.log2(1-pmf[b-1][:,:,np.newaxis]*psi[np.newaxis,:,:]*f_iu[i][np.newaxis,:,np.newaxis]*g_ikj[i][:,np.newaxis,:]/self.num)*pi[np.newaxis,:,np.newaxis]).sum(1)) for i in range(self.m)], axis=0))
            return out+log_pmf[a][:,:,np.newaxis]+log_sf[a+b-1][:,np.newaxis,:]
        def foo(a, b):
            if int(a)!=a or int(b)!=b or a>dia or b>dia or a<1 or b<1: return np.zeros((self.m, self.m, self.m))
            if a>b: return foo(b, a)
            #if b>a: return foo(b, a)
            #if independent: out = log_pmf[a][:,:,np.newaxis]+log_pmf[b][np.newaxis,:,:]+log_sf[a+b-1][:,np.newaxis,:]
            #else: out = nbr_to[b-1][a]+nbr_fr[a-1][b]+log_sf[a+b-1][:,np.newaxis,:]
            out = brg3(a, b)
            return np.exp2(out)
        out = [np.array([foo(j, i-j) for j in range(i+1)]) for i in x]
        if not apx:
            if not gcc_target: out = [o*gcc[np.newaxis,np.newaxis,:] for o in out]
            #if gcc_target: out = [o/gcc[np.newaxis,np.newaxis,:] for o in out]
            if not gcc_bridge: out = [o*gcc[np.newaxis,:,np.newaxis] for o in out]
            #if gcc_bridge: out = [o/gcc[np.newaxis,:,np.newaxis] for o in out]
            if not gcc_source: out = [o*gcc[:,np.newaxis,np.newaxis] for o in out]
        if mean_target: out = [self.get_mean(o, axis=3, gcc=gcc_target and not apx) for o in out]
        if mean_bridge: out = [self.get_mean(o, axis=2, gcc=gcc_bridge and not apx) for o in out]
        if mean_source: out = [self.get_mean(o, axis=1, gcc=gcc_source and not apx) for o in out]
        if mean_length: out = np.array([o.sum(0) for o in out])
        if len(x)==1: out = out[0]
        return out

    def _munp(self, n, mean_source=True, mean_target=True, apx=False, init_corrected=True):
        lmax = self.get_lmax()
        k = self.get_support(init=1).astype(float)
        p = self.pmf(k, mean_source=mean_source, mean_target=mean_target, gcc_source=True, gcc_target=True, apx=apx, init_corrected=init_corrected)
        return ((k**n)[(slice(None),)+(None,)*(p.ndim-1)]*p).sum(0)

    def stats(self, inverse=False, mean_source=False, mean_target=False, apx=False, init_corrected=True):
        if inverse: m1, m2 = self._munp(-1, mean_source, mean_target, apx, init_corrected), self._munp(-2, mean_source, mean_target, apx, init_corrected)
        else: m1, m2 = self._munp(1, mean_source, mean_target, apx, init_corrected), self._munp(2, mean_source, mean_target, apx, init_corrected)
        m2 = m2-m1**2
        return m1, m2

    def degree(self, mean_source=False, mean_target=True, total=True, gcc=True):
        if mean_source: assert(total)
        if total: out = self.psi
        elif gcc:
            phi = self.get_gcc()
            out = self.psi*(1-(1-phi[:,np.newaxis])*(1-phi[np.newaxis,:]))/phi[:,np.newaxis]
        else:
            phi = self.get_gcc()
            out = self.psi*(1-phi[np.newaxis,:])
        if mean_target: out = self.get_mean(out, axis=1)
        else: out = out*self.pi[np.newaxis,:]
        if mean_source: out = self.get_mean(out, axis=0)
        return out

    def closeness(self, mean=False, harmonic=False, source=False, gcc=True, apx=False, init_corrected=True):
        out, sigma = self.stats(inverse=harmonic, mean_source=not source, mean_target=source, apx=apx, init_corrected=init_corrected)
        if not harmonic: out = 1/out
        if not gcc and not apx: out = out*self.get_gcc()
        if mean: out = self.get_mean(out, gcc=gcc and not apx)
        return out

    def betweenness(self, mean=False, independent=False, gcc=True, apx=False, init_corrected=True):
        out = self.pmf_bridge(gcc_source=gcc, gcc_target=gcc, gcc_bridge=gcc, apx=apx, init_corrected=init_corrected, independent=independent).sum(axis=0)        
        if not gcc and not apx: out = out*self.get_gcc()
        if mean: out = self.get_mean(out, gcc=gcc and not apx)
        return out

    def pmf_joint(self, x=None, y=None, total=True, gcc=False, apx=False, init_corrected=True, independent=False):
        if x is None: x = self.get_support()
        if y is None: y = self.get_support()
        if not isiterable(x): x = [x]
        if not isiterable(y): y = [y]
        tmp = self.get_lmax(x)
        tmp = self.get_lmax(y)
        dia = self.get_lmax()
        pmf = self.pmf(total=False, apx=apx, init_corrected=init_corrected)
        cdf = pmf.cumsum(0)
        def foo(a, b):
            if int(a)!=a or int(b)!=b or a>dia or b>dia: return 0
            if independent: out = pmf[a][:,:,np.newaxis]*pmf[b][np.newaxis,:,:]
            else:
                p, q = max(a, b), min(a, b)
                if q==a: out = pmf[q][:,:,np.newaxis]*(pmf[p][np.newaxis,:,:]*((1-cdf[p-q]+pmf[p-q])[:,np.newaxis,:]) + pmf[p-q][:,np.newaxis,:]*((cdf[-1]-cdf[p])[np.newaxis,:,:]))
                else: out = pmf[q][np.newaxis,:,:]*(pmf[p][:,:,np.newaxis]*((1-cdf[p-q]+pmf[p-q])[:,np.newaxis,:]) + pmf[p-q][:,np.newaxis,:]*((cdf[-1]-cdf[p])[:,:,np.newaxis]))
            if gcc and not apx: out = out/(self.gcc_out[:,:,np.newaxis]*self.gcc_out[np.newaxis,:,:])
            return out
        out = np.array([[foo(i, j) for j in y] for i in x])
        if len(x)==len(y)==1: out = out[0,0]
        if total: out = self.get_mean(out, 3) #incorrect!
        return out

    def cdf_joint(self, x=None, y=None, total=True, gcc=False, apx=False, init_corrected=True, independent=False):
        if x is None: x = self.get_support()
        if y is None: y = self.get_support()
        if not isiterable(x): x = [x]
        if not isiterable(y): y = [y]
        tmp = self.get_lmax(x)
        tmp = self.get_lmax(y)
        dia = self.get_lmax()
        pmf = self.pmf_joint(total=total, gcc=gcc, apx=apx, init_corrected=init_corrected, independent=independent)
        cdf = np.zeros(pmf.shape)
        for i in range(1, dia+1):
            for j in range(1, dia+1): cdf[i][j] = cdf[i-1][j] + pmf[i][:j+1].sum(axis=0)
        out = np.array([[cdf[min(dia, int(i))][min(dia, int(j))] for j in y] for i in x])
        if len(x)==len(y)==1: out = out[0,0]
        return out

    def pmf_sum(self, k=None, total=True, gcc=False, apx=False, init_corrected=True, independent=False):
        pmf_joint = self.pmf_joint(self.get_support(k, 2), self.get_support(k, 2), total=False, gcc=gcc, apx=apx, init_corrected=init_corrected, independent=independent)
        if k is None: k = self.get_support(k, 2)
        if not isiterable(k): k = [k]
        out = np.array([np.array([pmf_joint[j][i-j] for j in range(i+1)]).sum(axis=0) if int(i)==i else np.zeros((self.m, self.m, self.m)) for i in k])
        if len(k)==1: out = out[0]
        if total: out = self.get_mean(out) #incorrect!
        return out

    def cdf_sum(self, k=None, total=True, gcc=False, apx=False, init_corrected=True, independent=False):        
        if k is None: k = self.get_support(k, 2)
        pmf = self.pmf_sum(self.get_support(k), total=total, gcc=gcc, apx=apx, init_corrected=init_corrected, independent=independent)
        cdf = pmf.cumsum(0)
        if not isiterable(k): k = [k]
        out = np.array([cdf[int(i)] for i in k])
        if len(k)==1: out = out[0]
        return out

    def cov(self, inverse=False, total=True, gcc=True, apx=False, init_corrected=True, independent=False):
        x = self.get_support(init=1)
        pmf = self.pmf(x, total=False, gcc=gcc, apx=apx)
        pmf_joint = self.pmf_joint(x, x, total=False, gcc=gcc, apx=apx, init_corrected=init_corrected, independent=independent)
        if inverse: n = -1
        else: n = 1
        x = x.astype(float)
        if len(x)==1:
            pmf = [pmf]
            pmf_joint = [[pmf_joint]]
        mu = np.array([x[i]**n*pmf[i] for i in range(len(x))]).sum(0)
        #use centered covariance estimation to prevent catastrophic cancellation and ensure numerical stability
        out = np.array([np.array([(x[i]**n-mu)[:,:,np.newaxis]*(x[j]**n-mu)[np.newaxis,:,:]*pmf_joint[i][j] for j in range(len(x))]).sum(0) for i in range(len(x))]).sum(0)
        if total: out = self.get_mean(out) #incorrect
        return out

    def paths_recursion(self, k, pmf=None, apx=False, init_corrected=True):
        m = self.m
        pi = self.pi
        psi = self.psi
        if pmf is None: pmf = self.pmf(np.arange(k+1), total=False, apx=apx, init_corrected=init_corrected)
        if int(k)!=k or k<1: return
        if k==1: return [np.ones((m, m))]
        else:
            val = self.paths_recursion(k-1, pmf)
            denom = pmf[:,:,k]
            idx = self.dia>=k #things should work stably till diameter
            factor = idx*(1-pmf[:,:,:k].sum(-1))
            factor[idx] = factor[idx]/denom[idx]
            curr = factor*((val[-1]*pmf[:,:,k-1])@np.diag(pi)@psi)
            curr[~idx] = val[-1][~idx]
            val.append(curr)
            return val

    def paths(self, k, apx=False, init_corrected=True):
        if isiterable(k): lmax = int(np.ceil(max(k)))
        else: lmax = int(np.ceil(k))
        output = self.paths_recursion(lmax, apx, init_corrected)
        if isiterable(k): output = np.dstack([output[int(i)-1] if int(i)==i and i>0 else np.zeros((self.m, self.m)) for i in k])
        else:
            if int(k)==k and k>0: output = output[int(k)-1]
            else: output = np.zeros((self.m, self.m))
        return output

    def eta(self, k, apx=False, init_corrected=True):
        paths = self.paths(k, apx, init_corrected)
        return paths, paths-1

    def eta_inv(self, k, apx=False, init_corrected=True):
        paths = self.paths(k, apx, init_corrected)
        m = self.m
        if isiterable(k):
            eta_i = [[[InversePoissonDistribution(paths[i,j,l]-1).stats() for j in range(m)] for i in range(m)] for l in range(paths.shape[-1])]
            mu = np.dstack([np.array([[j[0] for j in i] for i in l]) for l in eta_i])
            sigma = np.dstack([np.array([[j[1] for j in i] for i in l]) for l in eta_i])
        else:
            eta_i = [[InversePoissonDistribution(paths[i,j]-1).stats() for j in range(m)] for i in range(m)]
            mu = np.array([[j[0] for j in i] for i in eta_i])
            sigma = np.array([[j[1] for j in i] for i in eta_i])
        return mu, sigma

    def stats_closeness(self, harmonic=False, from_node=False, total=False, gcc=True, norm=True, apx=False, init_corrected=True):
        mu, sigma = self.stats(inverse=harmonic, total=False, gcc=gcc, apx=apx, init_corrected=init_corrected)
        cov = self.cov(inverse=harmonic, total=False, gcc=gcc, apx=apx, init_corrected=init_corrected)
        if norm: sigma = sigma/self.num + cov
        else: mu, sigma = mu*self.num, (sigma + cov*self.num)*self.num
        mu, sigma = self.get_mean(mu, axis=int(from_node), gcc=not from_node), self.get_mean(sigma, axis=int(from_node), gcc=not from_node) #closeness is typically distance "to" the nodes, and not "from"
        if total: mu, sigma = self.get_mean(mu, gcc=True), self.get_mean(sigma, gcc=True)
        return mu, sigma

    def stats_betweenness(self, apx=False, init_corrected=True):
        m = self.m
        pi = self.pi
        k = self.dia.max()+1
        pmf = self.pmf(np.arange(k+1), total=False, apx=apx, init_corrected=init_corrected)
        eta_mu, eta_sigma = self.eta(np.arange(1, k))
        eta_inv_mu, eta_inv_sigma = self.eta_inv(np.arange(2, k+1), apx)
        p_curr = 1
        mu, sigma = np.zeros((m, m, m)), np.zeros((m, m, m))
        for l_max in range(2, k+1):
            p_curr = p_curr - pmf[:,:,l_max-1]
            mu_curr, sigma_curr = np.zeros((m, m, m)), np.zeros((m, m, m))
            for l_low in range(1, l_max):
                l_upp = l_max-l_low
                mu_curr = mu_curr + (eta_mu[:,:,l_low-1]*pmf[:,:,l_low])[:,np.newaxis,:]*(eta_mu[:,:,l_upp-1]*pmf[:,:,l_upp]).T[np.newaxis,:,:]
                sigma_curr = sigma_curr + (eta_sigma[:,:,l_low-1]*pmf[:,:,l_low])[:,np.newaxis,:]*(eta_sigma[:,:,l_upp-1]*pmf[:,:,l_upp]).T[np.newaxis,:,:]
            mu = mu + mu_curr*(eta_inv_mu[:,:,l_max-2]*p_curr)[:,:,np.newaxis]
            sigma = sigma + sigma_curr*(eta_inv_sigma[:,:,l_max-2]*p_curr)[:,:,np.newaxis]
        mu = np.array([pi@mu[:,:,i]@pi for i in range(m)])
        sigma = np.array([pi@sigma[:,:,i]@pi for i in range(m)])
        return mu, sigma
