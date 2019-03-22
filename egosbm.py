from collections.abc import MutableSequence
import numpy as np

class EgocentricSBM(MutableSequence):
    
    def __init__(self, num_blocks=tuple(), dim_names=None, model_name='egocentric_sbm'):
        
        if not self.iterable(num_blocks): num_blocks = (num_blocks,)
        num_blocks = tuple(num_blocks)
        if not all([isinstance(b, int) and b>1 for b in num_blocks]): raise ValueError('number of blocks must be an integer greater than 1')
        self._precision = 8
        self._curridx = 0
        self._currdim = 0
        self.ndim = len(num_blocks)
        self.shape = num_blocks
        self.name = str(model_name)
        self.dims = tuple()
        self.dimsdict = dict()
        self.set_dims(dim_names)
        self.pi = tuple([(1/b,)*b for b in num_blocks])
        self.rho = tuple([(1/b,)*b for b in num_blocks])
        self.omega = tuple([(1.,)*b for b in num_blocks])
        self.check_consistency()
        
    @property
    def precision(self): return self._precision
    
    @precision.setter
    def precision(self, value):
        from warnings import warn
        warn('setting "precision" can affect model consistency checks', RuntimeWarning)
        self._precision = value
        
    def __str__(self):
        out = ['name:\t'+self.name]
        out.append('ndim:\t'+str(self.ndim))
        out.append('shape:\t'+str(self.shape))
        out.append('|dims|\t|blocks|')
        out += [str(x[0])+'\t'+str(x[1]) for x in self.dims]
        out.append('pi:\t'+str(self.pi))
        out.append('rho:\t'+str(self.rho))
        out.append('omega:\t'+str(self.omega))
        return '\n'.join(out)
    
    def __len__(self): return self.ndim
    
    def __copy__(self):
        out = type(self)(self.shape, self.dims, self.name)
        out.pi = self.pi
        out.rho = self.rho
        out.omega = self.omega
        out._precision = self._precision
        out._currdim = self._currdim
        return out
    
    def copy(self): return self.__copy__()
    
    def sort(self, inplace=True, reverse=True, blocks=True, dims=True):
        idx_blocks = list()
        if blocks or dims:
            # sorts by pi, followed by omega to break ties, followed by rho to break ties
            for i in range(self.ndim):
                idx_blocks.append(tuple(sorted(zip(range(self.shape[i]), zip(self.pi[i], self.omega[i], self.rho[i])), key=lambda t: t[1], reverse=reverse)))
        idx_dims = list()
        if dims:
            from itertools import chain
            # sorts by number of blocks, followed by atts of block with largest pi, and so on
            temp = [(self.shape[i],) + tuple(chain(*[x[1] for x in idx_blocks[i]])) for i in range(self.ndim)]
            idx_dims = sorted(zip(range(self.ndim), temp), key=lambda t: t[1], reverse=reverse)
        if inplace:
            if dims:
                self.shape = tuple([x[1][0] for x in idx_dims])
                if blocks: temp = [(self.dims[x[0]][0], [self.dims[x[0]][1][y[0]] for y in idx_blocks[x[0]]]) for x in idx_dims]
                else: temp = [self.dims[x[0]] for x in idx_dims]
                self.set_dims(temp)
            if blocks:
                self.pi = tuple([tuple([y[1][0] for y in idx_blocks[x[0]]]) for x in idx_dims])
                self.omega = tuple([tuple([y[1][1] for y in idx_blocks[x[0]]]) for x in idx_dims])
                self.rho = tuple([tuple([y[1][2] for y in idx_blocks[x[0]]]) for x in idx_dims])
        else:
            if dims: idx_dims = [x[0] for x in idx_dims]
            if blocks: idx_blocks = [[x[0] for x in y] for y in idx_blocks]
            return idx_dims, idx_blocks
    
    def get_key(self, key):
        if key in self.dimsdict:
            key = self.dimsdict[key]
        else:
            if isinstance(key, int):
                if key<0: key += self.ndim
                if key>=0 and key<self.ndim: pass
                else: raise IndexError('sbm index out of range; must be less than ndims')
            else: raise KeyError('sbm key "%s" not found; must be dimension name'%str(key))
        return key
    
    def insert(self, key, num_blocks, block_names=None):
        if not (isinstance(num_blocks, int) and num_blocks>1): raise ValueError('number of blocks must be an integer greater than 1')
        if key in self.dimsdict:
            from warnings import warn
            warn('this will replace dimension "%s" which already exists'%key)
            key = self.dimsdict[key]
            old_shape = self.shape
            try:
                temp = list(self.shape)
                temp[key] = num_blocks
                self.shape = tuple(temp)
                temp = list(self.dims)
                if block_names is None:
                    temp[key] = temp[key][0]
                    self.set_dims(temp)
                elif block_names == 'old':
                    if len(temp[key][1])!=num_blocks: raise ValueError('cannot retain old block names since new shape (%d) does not match the old shape (%d)'%(len(temp[key][1]), num_blocks))
                else:
                    temp[key] = (temp[key][0],  block_names)
                    self.set_dims(temp)
            except Exception as err:
                self.shape = old_shape
                raise err
            temp = list(self.omega)
            mean_omega = self.mean_omega()
            if len(mean_omega)>0: mean_omega = mean_omega[0]
            else: mean_omega = 0.0
            temp[key] = (mean_omega,)*num_blocks
            self.omega = tuple(temp)
            temp = list(self.pi)
            temp[key] = (1/num_blocks,)*num_blocks
            self.pi = tuple(temp)
            temp = list(self.rho)
            temp[key] = (1/num_blocks,)*num_blocks
            self.rho = tuple(temp)
        elif isinstance(key, int) or isinstance(key, str):
            if isinstance(key, str):
                keyname = key
                key = self.ndim
            else: keyname = None
            old_shape = self.shape
            try:                
                temp = list(self.shape)
                temp.insert(key, num_blocks)
                self.shape = tuple(temp)            
                temp = list(self.dims)
                if block_names is None: temp.insert(key, keyname)
                else: temp.insert(key, (keyname,  block_names))
                self.ndim += 1
                self.set_dims(temp)
            except Exception as err:
                self.ndim -=1
                self.shape = old_shape
                raise err
            temp = list(self.omega)
            mean_omega = self.mean_omega()
            if len(mean_omega)>0: mean_omega = mean_omega[0]
            else: mean_omega = 0.0
            temp.insert(key, (mean_omega,)*num_blocks)
            self.omega = tuple(temp)
            temp = list(self.pi)
            temp.insert(key, (1/num_blocks,)*num_blocks)
            self.pi = tuple(temp)
            temp = list(self.rho)
            temp.insert(key, (1/num_blocks,)*num_blocks)
            self.rho = tuple(temp)
        else: raise ValueError('either enter an integer index (dimension number) or a string key (dimension name)')
    
    def __getitem__(self, key):
        key = self.get_key(key)
        return {'index':key,
                'name':self.dims[key][0], 
                'blocks':self.dims[key][1], 
                'pi':self.pi[key], 
                'rho':self.rho[key], 
                'omega':self.omega[key]}
    
    def __setitem__(self, key, value):
        if not self.iterable(key):
            if isinstance(value, dict):
                if len(value)==1:
                    key = (key, list(value.keys())[0])
                    value = list(value.values())[0]
                else: raise ValueError('you can set only one attribute at a time')
            elif self.iterable(value):
                if len(value)==2:
                    key = (key, value[0])
                    value = value[1]
                else: raise ValueError('you can set only one attribute at a time')
            else: raise ValueError('provide a "(p, v)" tuple as value and "d" as key to set property "p" of dimension "d" to value "v"')
        if len(key)==2:
            k = self.get_key(key[0])
            if key[1]=='pi':
                curr_pi = list(self.pi)
                curr_pi[k] = value
                self.set_pi(curr_pi)
            elif key[1]=='rho':
                curr_rho = list(self.rho)
                curr_rho[k] = value
                self.set_rho(curr_rho)
            elif key[1]=='omega':
                curr_omega = list(self.omega)
                curr_omega[k] = value
                self.set_omega(curr_omega)
            elif key[1]=='blocks':
                curr_dims = list(self.dims)
                curr_dims[k] = list(curr_dims[k])
                curr_dims[k][1] = value
                self.set_dims(curr_dims)
            elif key[1]=='name':
                curr_dims = list(self.dims)
                curr_dims[k] = list(curr_dims[k])
                curr_dims[k][0] = value
                self.set_dims(curr_dims)
            elif key[1]=='index': raise ValueError('cannot set dim index; use pop() and insert() methods to reorder dims')
            else: raise ValueError('no such property "%s"'%str(key[1]))
        else: raise ValueError('provide a "(d, p)" tuple as key and "v" as value to set property "p" of dimension "d" to value "v"')        
            
    def __delitem__(self, key):
        key = self.get_key(key)
        self.ndim -= 1
        temp = list(self.shape)
        temp.pop(key)
        self.shape = tuple(temp)
        temp = list(self.pi)
        pi = temp.pop(key)
        self.pi = tuple(temp)
        temp = list(self.rho)
        rho = temp.pop(key)
        self.rho = tuple(temp)
        temp = list(self.omega)
        omega = temp.pop(key)
        self.omega = tuple(temp)
        temp = list(self.dims)
        dims = temp.pop(key)
        self.dims = tuple(temp)
        self.dimsdict = dict(zip(temp, range(self.ndim)))
        return {'index':key,
                'name':dims[0], 
                'blocks':dims[1], 
                'pi':pi, 
                'rho':rho, 
                'omega':omega}
        
    def __contains__(self, key): return key in self.dimsdict
    
    def __iter__(self): return self
    
    def __next__(self):
        if self._curridx>=self.ndim:
            self._curridx = 0
            raise StopIteration
        else:
            self._curridx += 1
            return self[self._curridx-1]
        
    def __eq__(self, other):
        x = self.copy()
        y = other.copy()
        x.sort()
        y.sort()
        return x.pi==y.pi and x.rho==y.rho and x.omega==y.omega
     
    def __lt__(self, other):
        if self.ndim < other.ndim:
            x = self.copy()
            y = other.copy()
            x.sort()
            y.sort()
            i = 0
            j = 0
            while i<self.ndim:
                found = False
                while j<other.ndim:
                    if x.pi[i]==y.pi[j] and x.rho[i]==y.rho[j] and x.omega[i]==y.omega[j]:
                        found = True
                        break
                    else: j+=1
                if not found: return False
                else:
                    i+=1
            return True 
        else: return False
        
    def __le__(self, other):
        if self==other: return True
        else: return self<other
        
    def ishomophilous(self): return tuple([tuple([u<v for u,v in zip(i,j)]) for i,j in zip(self.pi, self.rho)])
    def isheterophilous(self): return tuple([tuple([u>v for u,v in zip(i,j)]) for i,j in zip(self.pi, self.rho)])
    def isambiphilous(self): return tuple([tuple([u==v for u,v in zip(i,j)]) for i,j in zip(self.pi, self.rho)])
    
    def iterable(self, obj):
        if isinstance(obj, str): return False
        try:
            iter(obj)
            return True
        except TypeError: return False
    
    def sum_pi(self, pi=None, approx=False):
        if pi is None: pi = self.pi
        out = [sum(p) for p in pi]
        if approx: out = [round(x, self.precision) for x in out]
        return out
    
    def sum_rho(self, rho=None, approx=False):
        if rho is None: rho = self.rho
        out = [sum(r) for r in rho]
        if approx: out = [round(x, self.precision) for x in out]
        return out
    
    def dev_pi(self, pi=None, approx=False):
        if pi is None: pi = self.pi
        out = [sum([i**2 for i in p]) for p in pi]
        if approx: out = [round(x, self.precision) for x in out]
        return out
    
    def mean_rho(self, rho=None, pi=None, approx=False):
        if pi is None: pi = self.pi
        if rho is None: rho = self.rho
        out = [sum([m*n for m, n in zip(r, p)]) for r, p in zip(rho, pi)]
        if approx: out = [round(x, self.precision) for x in out]
        return out
    
    def mean_omega(self, omega=None, pi=None, approx=False):
        if pi is None: pi = self.pi
        if omega is None: omega = self.omega
        out = [sum([m*n for m, n in zip(o, p)]) for o, p in zip(omega, pi)]
        if approx: out = [round(x, self.precision) for x in out]
        return out
    
    def mean_homophily(self, rho=None, approx=False): return self.sum_rho(rho, approx)
    
    def mean_heterophily(self, rho=None, pi=None, approx=False):
        if pi is None: pi = self.pi
        if rho is None: rho = self.rho
        out = [sum([(1-m)*n/(1-n) for m, n in zip(r, p)]) for r, p in zip(rho, pi)]
        if approx: out = [round(x, self.precision) for x in out]
        return out
    
    def mean_homoffinity(self, omega=None, rho=None, approx=False):
        if rho is None: rho = self.rho
        if omega is None: omega = self.omega
        out = [sum([m*n for m, n in zip(o, r)]) for o, r in zip(omega, rho)]
        if approx: out = [round(x, self.precision) for x in out]
        return out
    
    def mean_heteroffinity(self, omega=None, rho=None, pi=None, approx=False):
        if pi is None: pi = self.pi
        if rho is None: rho = self.rho
        if omega is None: omega = self.omega
        out = [sum([(1-m)*n*w/(1-n) for w, m, n in zip(o, r, p)]) for o, r, p in zip(omega, rho, pi)]
        if approx: out = [round(x, self.precision) for x in out]
        return out

    def check_consistency(self, pi=None, omega=None, rho=None):
        if pi is None: pi = self.pi
        if omega is None: omega = self.omega
        if rho is None: rho = self.rho
        #pi tests
        if not all([all([0<i<1 for i in p])for p in pi]): raise ValueError('pi must be between 0 and 1 (exclusive)')
        if not all([p==1 for p in self.sum_pi(pi=pi, approx=True)]): raise ValueError('pi must sum up to 1')
        #omega tests
        if not all([all([0<=i for i in o])for o in omega]): raise ValueError('omega must be >=0')
        mean_omega = self.mean_omega(omega=omega, approx=True)
        for i in range(1, len(mean_omega)):
            if not mean_omega[i] == mean_omega[0]: raise ValueError('mean omega must remain same')
        #rho tests
        if not all([all([0<=i<=1 for i in r])for r in rho]): raise ValueError('rho must be between 0 and 1 (inclusive)')

    def get_dims(self):
        from itertools import product
        dims_flat = [[(dim[0]+':'+str(d)) for d in dim[1]] for dim in self.dims]
        return tuple([','.join(i) for i in product(*dims_flat)])
    
    def get_nextdim(self):
        while str(self._currdim) in self.dimsdict:
            self._currdim += 1
        self._currdim += 1
        return str(self._currdim-1)
    
    def set_dims(self, dims=None):
        if dims is None: dims = [None for i in range(self.ndim)]
        if not self.iterable(dims): raise ValueError('provide all dimension names')
        dims = list(dims)
        if len(dims)!=self.ndim: raise ValueError('provide names for each dimension')
        dim_set = []
        for i in range(self.ndim):
            if self.iterable(dims[i]):
                found = False
                if len(dims[i])==2:
                    if (not self.iterable(dims[i][0])) and self.iterable(dims[i][1]):
                        if len(dims[i][1])==self.shape[i] and all([not self.iterable(j) for j in dims[i][1]]):
                            if len(set(dims[i][1]))!=self.shape[i]: raise ValueError('provide *unique* names for each block of dimension %d'%i)
                            else:
                                found = True
                                if dims[i][0] is None: dims[i] = (self.get_nextdim(), tuple(dims[i][1]))
                                else: dims[i] = (dims[i][0], tuple(dims[i][1]))
                        else: raise ValueError('provide names for each block of dimension %d'%i)
                if (not found) and len(dims[i])==self.shape[i] and all([not self.iterable(j) for j in dims[i]]):
                    if len(set(dims[i]))!=self.shape[i]: raise ValueError('provide *unique* names for each block of dimension %d'%i)
                    else:
                        found = True
                        dims[i] = (str(i), tuple(dims[i]))
                if not found: raise ValueError('provide names for each block of dimension %d'%i)
            else:
                if dims[i] is None: dims[i] = self.get_nextdim()
                dims[i] = (dims[i], tuple([str(j) for j in range(self.shape[i])]))
            if dims[i][0] in dim_set: raise ValueError('provide *unique* name for each dimension; clash on "%s"'%str(dims[i][0]))
            else: dim_set.append(dims[i][0])        
        if any([not isinstance(dim, str) for dim in dim_set]): raise TypeError('dimension names must be strings')
        self.dims = tuple(dims)
        self.dimsdict = dict(zip(dim_set, range(self.ndim)))
        
    def get_random_pi(self, pi, omega, min_pi=0., max_pi=1., max_iters=100):
        from random import triangular
        i = 0
        min_pi = min(pi, min_pi)
        max_pi = max(pi, max_pi)
        assert(0<=min_pi<=pi<=max_pi<=1)
        pair = tuple()
        for i in range(len(omega)):
            for j in range(i+1, len(omega)):
                if omega[i]!=omega[j]:
                    pair = (i, j)
                    break
            if pair: break
        if not pair: pair = (len(omega)-2, len(omega)-1)
        while i<max_iters:
            p = [0. for i in range(len(omega))]
            for i in range(len(omega)):
                if i not in pair: p[i] = triangular(min_pi, max_pi, pi)
            t = (1 - sum(p))
            mo = self.mean_omega()[0]
            if omega[pair[0]]!=omega[pair[1]]: temp1 = ((mo - sum([o*q for (o, q) in zip(omega, p)])) - omega[pair[0]]*t)/(omega[pair[1]]-omega[pair[0]])
            else: temp1 = triangular(min_pi, max_pi, pi)
            temp2 = t - temp1
            if min_pi<=temp1<=max_pi and min_pi<=temp2<=max_pi:
                p[pair[0]] = temp2
                p[pair[1]] = temp1
                return tuple(p)
            else: i+=1
        raise RuntimeError('max iters reached; appropriate pis not found')
    
    def get_random_omega(self, omega, pi, min_omega=0, max_omega=100, max_iters=100):
        from random import triangular
        i = 0
        min_omega = min(omega, min_omega)
        max_omega = max(omega, max_omega)
        assert(0<=min_omega<=omega<=max_omega)
        while i<max_iters:
            om = list()
            for i in range(len(pi)-1):
                om.append(triangular(min_omega, max_omega, omega))
            temp = (omega - sum([o*p for (o, p) in zip(om, pi[:-1])]))/pi[-1]
            if min_omega<=temp<=max_omega:
                om.append(temp)
                return tuple(om)
            else: i+=1
        raise RuntimeError('max iters reached; appropriate omegas not found')
        
    def get_random_rho(self, rho, n, min_rho=0., max_rho=1.):
        from random import triangular
        min_rho = min(rho, min_rho)
        max_rho = max(rho, max_rho)
        assert(0<=min_rho<=rho<=max_rho<=1)
        return tuple([triangular(min_rho, max_rho, rho) for i in range(n)])

    def set_pi(self, pi=None, random=True):
        if pi is None:
            if random: pi = tuple([self.get_random_pi(1/self.shape[i], self.omega[i]) for i in range(self.ndim)])
            else: pi = tuple([(1/b,)*b for b in self.shape])
        if not self.iterable(pi):
            if random: pi = tuple([self.get_random_pi(pi, self.omega[i]) for i in range(self.ndim)])
            else: pi = tuple([(float(pi),)*b for b in self.shape])
        pi = list(pi)
        if all([not self.iterable(p) for p in pi]) and self.ndim==1: pi = [pi]
        if len(pi)!=self.ndim: raise ValueError('provide pi for every dimension')
        for i in range(self.ndim):
            if not self.iterable(pi[i]):
                if random: pi[i] = self.get_random_pi(pi[i], self.omega[i])
                else: pi[i] = (float(pi[i]),)*self.shape[i]
            if len(pi[i])!=self.shape[i]: raise ValueError('provide pi for every block of dimension %d'%i)
            else: pi[i] = tuple(pi[i])
        pi = tuple(pi)
        self.check_consistency(pi=pi)
        self.pi = pi
                
    def set_omega(self, omega=1., random=True):
        if not self.iterable(omega):
            if random: omega = tuple([self.get_random_omega(omega, self.pi[i]) for i in range(self.ndim)])
            else: omega = tuple([(float(omega),)*b for b in self.shape])
        omega = list(omega)
        if all([not self.iterable(o) for o in omega]) and self.ndim==1: omega = [omega]
        if len(omega)!=self.ndim: raise ValueError('provide omega for every dimension')
        for i in range(self.ndim):
            if not self.iterable(omega[i]):
                if random: omega[i] = self.get_random_omega(omega[i], self.pi[i])
                else: omega[i] = (float(omega[i]),)*self.shape[i]
            if len(omega[i])!=self.shape[i]: raise ValueError('provide omega for every block of dimension %d'%i)
            else: omega[i] = tuple(omega[i])
        omega = tuple(omega)
        self.check_consistency(omega=omega)
        self.omega = omega
        
    def set_rho(self, rho=None, random=True):
        if rho is None:
            if random: rho = tuple([self.get_random_rho(1/i, i) for i in self.shape])
            else: rho = tuple([(1/b,)*b for b in self.shape])
        if not self.iterable(rho):
            if random: rho = tuple([self.get_random_rho(rho, i) for i in self.shape])
            else: rho = tuple([(float(rho),)*b for b in self.shape])
        rho = list(rho)
        if all([not self.iterable(r) for r in rho]) and self.ndim==1: rho = [rho]
        if len(rho)!=self.ndim: raise ValueError('provide rho for every dimension')
        for i in range(self.ndim):
            if not self.iterable(rho[i]):
                if random: rho[i] = self.get_random_rho(rho[i], self.shape[i])
                else: rho[i] = (float(rho[i]),)*self.shape[i]
            if len(rho[i])!=self.shape[i]: raise ValueError('provide rho for every block of dimension %d'%i)
            else: rho[i] = tuple(rho[i])
        rho = tuple(rho)
        self.check_consistency(rho=rho)
        self.rho = rho  
    
    def enforce_homophily(self):
        rho = [[0 for j in range(self.shape[i])] for i in range(self.ndim)]
        for i in range(self.ndim):
            for j in range(self.shape[i]):
                min_rho = self.pi[i][j]
                rho[i][j] = self.get_random_rho((min_rho+1)/2, 1, min_rho, 1)[0]
        self.set_rho(rho)
        
    def enforce_heterophily(self):
        rho = [[0 for j in range(self.shape[i])] for i in range(self.ndim)]
        for i in range(self.ndim):
            for j in range(self.shape[i]):
                max_rho = self.pi[i][j]
                rho[i][j] = self.get_random_rho(max_rho/2, 1, 0, max_rho)[0]
        self.set_rho(rho)
        
    def enforce_ambiphily(self): self.set_rho(self.pi)
    
    def get_model(self, model_type='full', model_name='sbm'): return self.StochasticBlockModel(self, model_type, True, model_name)
    
    class StochasticBlockModel():
        
        def __init__(self, egocentric_sbm, model_type='full', directed=True, model_name='sbm'):
            
            if model_type not in ['full', 'pp', 'ppcollapsed']: raise ValueError('invalid model type "%s"'%model_type)
            self.type = model_type
            self.name = model_name
            self.ndim = egocentric_sbm.ndim
            self.shape = egocentric_sbm.shape
            self.dims = egocentric_sbm.get_dims()
            self.pi = [pi for pi in egocentric_sbm.pi]
            self.meanomega = egocentric_sbm.mean_omega()[0]
            self._precision = egocentric_sbm.precision
            
            if model_type=='full':
                self.params = [[(o, r/p, (1-r)/(1-p)) for r, o, p in zip(egocentric_sbm.rho[i], egocentric_sbm.omega[i], egocentric_sbm.pi[i])] for i in range(egocentric_sbm.ndim)]
                self.psi = [(np.diag([x[0] for x in item]), np.diag([x[1]-x[2] for x in item])+np.vstack([x[2]*np.ones(len(item)) for x in item])) for item in self.params]
            elif model_type=='pp': #planted-partition model
                self.params = (self.meanomega, list(zip(egocentric_sbm.mean_homophily(), egocentric_sbm.mean_heterophily())))
                self.psi = [(self.params[0]*np.eye(egocentric_sbm.shape[i]), (self.params[1][i][0]-self.params[1][i][1])*np.eye(egocentric_sbm.shape[i])+self.params[1][i][1]*np.ones([egocentric_sbm.shape[i], egocentric_sbm.shape[i]])) for i in range(egocentric_sbm.ndim)]
            elif model_type=='ppcollapsed':
                self.params = list(zip(egocentric_sbm.mean_homoffinity(), egocentric_sbm.mean_heteroffinity()))
                self.psi = [(self.params[i][0]-self.params[i][1])*np.eye(egocentric_sbm.shape[i])+self.params[i][1]*np.ones([egocentric_sbm.shape[i], egocentric_sbm.shape[i]]) for i in range(egocentric_sbm.ndim)]
            
            temp = self.get_psi(directed=True)
            self.directed = directed and (temp!=temp.transpose()).any()
                
        @property
        def precision(self): return self._precision

        @precision.setter
        def precision(self, value):
            from warnings import warn
            warn('setting "precision" can affect model consistency checks', RuntimeWarning)
            self._precision = value
            
        def __str__(self):
            out = ['name:\t'+self.name]
            out.append('type:\t'+str(self.type))
            out.append('ndim:\t'+str(self.ndim))
            out.append('shape:\t'+str(self.shape))
            out.append('pi:\t'+str(self.pi))
            out.append('params:\t'+str(self.params))
            return '\n'.join(out)
    
        def __len__(self): return self.get_shape()
            
        def kron(self, list_of_mats):
                a = np.array(list_of_mats[0])
                for i in range(1, len(list_of_mats)):
                    a = np.kron(a, np.array(list_of_mats[i]))
                return a
            
        def get_shape(self): return int(self.kron(self.shape))
        
        def get_pi(self, pi=None):
            if pi is None: return self.kron(self.pi)
            if isinstance(pi, str) and pi=='uni': return (1/self.get_shape())*np.ones([self.get_shape()])
            if len(pi)==self.ndim:
                for i in self.ndim:
                    if len(pi[i])!=self.shape[i]: raise ValueError('provide pi for all blocks and in proper dimension order')
                pi = self.kron(pi)
            if len(pi)!=self.get_shape(): raise ValueError('provide full or factorised pi')
            if not all([0<=p<=1 for p in pi]): raise ValueError('pi must be between 0 and 1 (inclusive)')
            if round(sum(pi), self.precision)!=1: raise ValueError('pi must sum up to 1')
            return np.array(pi)
            
        def get_psi(self, directed=None, log_ratio=False):
            multiplier = 1/(self.meanomega**(self.ndim-1))
            if self.type=='ppcollapsed': out = multiplier*self.kron(self.psi)
            elif self.type=='pp': out = multiplier*np.matmul(self.kron([x[0] for x in self.psi]), self.kron([x[1] for x in self.psi]))
            elif self.type=='full': out =  multiplier*np.matmul(self.kron([x[0] for x in self.psi]), self.kron([x[1] for x in self.psi]))
            if directed is None: directed = self.directed
            if not directed:
                if self.directed:
                    from warnings import warn
                    warn('symmetrising the stochastic block matrix', RuntimeWarning)
                out = (out + out.transpose())/2
            if log_ratio: out = np.log2(out) - np.hstack([np.log2(np.diag(out))[:,np.newaxis]]*self.get_shape())
            return out
        
        def generate_people(self, n=100, pi=None): return np.random.multinomial(1, self.get_pi(pi), n)
        
        def generate_network(self, people, directed=None):
            n, d = people.shape
            people = np.array(np.array(people, dtype=bool), dtype=int)
            factor = 1/(n-1) #correction to keep graph simple
            if d!=self.get_shape(): raise ValueError('number of columns must correspond to full number of blocks')
            if not (all([i==1 for i in people.sum(1)]) and all([1 in i for i in people])): raise ValueError('expected single-memberships only')
            p = np.matmul(np.matmul(people, factor*self.get_psi()), people.transpose())
            p[p>1] = 1.
            np.fill_diagonal(p, 0)
            a = np.random.binomial(1, p)
            if directed is None: directed = self.directed
            if not directed:
                if self.directed:
                    from warnings import warn
                    warn('generating an undirected network from a directed SBM', RuntimeWarning)
                a = np.triu(a)
                a = a + a.transpose()
            return (a, p)
        
        def generate_data(self, n=100, pi=None, directed=False, name='network_data'):
            z = self.generate_people(n, pi)
            a, p = self.generate_network(z, directed)
            return NetworkData(a, z, self.dims, p, name)
        
        def eigvals_pipsi(self, pi=None, directed=None, real=True):
            pi = self.get_pi(pi)
            if real: return sorted(np.real(np.linalg.eigvals(np.matmul(np.diag(pi), self.get_psi(directed)))), reverse=True)
            else: return sorted(np.linalg.eigvals(np.matmul(np.diag(pi), self.get_psi(directed))), reverse=True)
        
        def eigvals_pipsi_theoretical(self):
            pi = 1/self.get_shape()
            if self.type=='full': raise RuntimeError('cannot compute theoretical eigenvalues for a "full" model')
            elif self.type=='pp':
                factor = pi*self.meanomega
                eigs = sorted(tuple(factor*self.kron([(h[0]+h[1]*(s-1),)+(h[0]-h[1],)*(s-1) for (h, s) in zip(self.params[1], self.shape)])), reverse=True)
            elif self.type=='ppcollapsed':
                factor = pi/(self.meanomega**(self.ndim-1))
                eigs = sorted(tuple(factor*self.kron([(h[0]+h[1]*(s-1),)+(h[0]-h[1],)*(s-1) for (h, s) in zip(self.params, self.shape)])), reverse=True)
            return eigs
        
        def homoffinity_out(self, pi=None, directed=None, log_ratio=False): return np.diag(self.get_psi(directed, log_ratio))*self.get_pi(pi)
        def homoffinity_in(self, pi=None, directed=None, log_ratio=False): return self.homoffinity_out(pi, directed, log_ratio)
        def homoffinity(self, pi=None, directed=None, log_ratio=False): return self.homoffinity_out(pi, directed, log_ratio)
            
        def heteroffinity_out(self, pi=None, directed=None, log_ratio=False): return self.affinity_out(pi, directed, log_ratio)-self.homoffinity_out(pi, directed, log_ratio)
        def heteroffinity_in(self, pi=None, directed=None, log_ratio=False): return self.affinity_in(pi, directed, log_ratio)-self.homoffinity_in(pi, directed, log_ratio)
        def heteroffinity(self, pi=None, directed=None, log_ratio=False): return self.affinity(pi, directed, log_ratio)-self.homoffinity(pi, directed, log_ratio)
        
        def affinity_out(self, pi=None, directed=None, log_ratio=False): return np.dot(self.get_psi(directed, log_ratio), self.get_pi(pi))
        def affinity_in(self, pi=None, directed=None, log_ratio=False): return np.dot(self.get_psi(directed, log_ratio).transpose(), self.get_pi(pi))
        def affinity(self, pi=None, directed=None, log_ratio=False):
            if self.directed and (directed or directed is None): raise RuntimeError('affinity is ambiguous for directed SBMs; use affinity_in() or affinity_out()')
            else: return self.affinity_out(pi, directed, log_ratio)
        
        def mean_homoffinity_out(self, pi=None, directed=None, log_ratio=False): return (self.homoffinity_out(pi, directed, log_ratio)*self.get_pi(pi)).sum()
        def mean_homoffinity_in(self, pi=None, directed=None, log_ratio=False): return (self.homoffinity_in(pi, directed, log_ratio)*self.get_pi(pi)).sum()
        def mean_homoffinity(self, pi=None, directed=None, log_ratio=False): return (self.homoffinity(pi, directed, log_ratio)*self.get_pi(pi)).sum()
        
        def mean_heteroffinity_out(self, pi=None, directed=None, log_ratio=False): return (self.heteroffinity_out(pi, directed, log_ratio)*self.get_pi(pi)).sum()
        def mean_heteroffinity_in(self, pi=None, directed=None, log_ratio=False): return (self.heteroffinity_in(pi, directed, log_ratio)*self.get_pi(pi)).sum()
        def mean_heteroffinity(self, pi=None, directed=None, log_ratio=False): return (self.heteroffinity_out(pi, directed, log_ratio)*self.get_pi(pi)).sum()
        
        def mean_affinity_out(self, pi=None, directed=None, log_ratio=False): return (self.affinity_out(pi, directed, log_ratio)*self.get_pi(pi)).sum()
        def mean_affinity_in(self, pi=None, directed=None, log_ratio=False): return (self.affinity_in(pi, directed, log_ratio)*self.get_pi(pi)).sum()
        def mean_affinity(self, pi=None, directed=None, log_ratio=False): return (self.affinity_out(pi, directed, log_ratio)*self.get_pi(pi)).sum()
        
        def floyd_warshall(self, pw_dist_matrix, paths=True):
            if (pw_dist_matrix<0).any():
                from warnings import warn
                warn('negative pairwise distances can lead to negative path lengths')
            a, b = pw_dist_matrix.shape
            if a!=b: raise ValueError('expected square pairwise distance matrix')
            dist = pw_dist_matrix.copy()
            idx = np.array([[i]*a for i in range(a)])
            for i in range(a):
                dist[i,i] = pw_dist_matrix[i,i]
                idx[:,i] = i
            for k in range(a):
                for i in range(a):
                    for j in range(a):
                        if dist[i,j] > dist[i,k]+dist[k,j]:
                            dist[i,j] = dist[i,k]+dist[k,j]
                            idx[i,j] = idx[k,j]
                        if i==j and dist[i,j]<0: warn('negative cycle involving community "%d" found'%i)
            if paths:
                paths= [[[i] for j in range(a)] for i in range(a)]
                for i in range(a):
                    for j in range(a):
                        if i==j and idx[i,i]==i: paths[i][i].append(i)
                        else:
                            k = i
                            while k!=j:
                                k = idx[k,j]
                                paths[i][j].append(k)
                return (dist, tuple(paths))
            else: return dist
            
        def dis2met(self, distance_matrix, metric_type='metric', positivity=True):
            if metric_type is None: return distance_matrix            
            def metric_constraints(metric_type):
                if metric_type in ['metric', 'pseudometric']: return dict(identity=True, symmetry=True, triangle=True)
                elif metric_type=='metametric': return dict(identity=False, symmetry=True, triangle=True)
                elif metric_type=='quasimetric': return dict(identity=True, symmetry=False, triangle=True)
                elif metric_type=='semimetric': return dict(identity=True, symmetry=True, triangle=False)
                elif metric_type in ['premetric', 'prametric', 'pseudosemimetric']: return dict(identity=True, symmetry=False, triangle=False)
                elif metric_type=='pseudometametric': return dict(identity=False, symmetry=True, triangle=False)
                elif metric_type in ['hemimetric', 'pseudoquasimetric']: return dict(identity=False, symmetry=False, triangle=True)
                elif metric_type is None: return dict(identity=False, symmetry=False, triangle=False)
                else: raise ValueError ('metric type "%s" not found'%str(metric_type))
            a, b = distance_matrix.shape
            if a!=b: raise ValueError('expected square metric matrix')
            metric = distance_matrix.copy()
            metric[np.isnan(metric)] = np.nanmin(metric)
            constraints = metric_constraints(metric_type)
            if constraints['identity']: metric = metric - np.hstack([np.diag(metric)[:,np.newaxis]]*self.get_shape()) #indiscernibility of identiticals
            if (metric<0).any(): temp = metric.min()
            else: temp = 0.0
            metric = metric - temp #for positivity
            paths = [[[0, 0]]*a]*a
            if constraints['symmetry']: metric = (metric + metric.transpose())/2 #symmetry
            if constraints['triangle']: metric, paths = self.floyd_warshall(metric) #triangle inequality
            if not positivity: metric = metric + np.array([[temp*(len(paths[i][j])-1) for j in range(len(paths[i]))] for i in range(len(paths))])
            return metric
        
        def ind2com(self, matrix, pi=None):
            a, b = matrix.shape
            if a!=b or a!=self.get_shape(): raise ValueError('expected square matrix of size %d'%self.get_shape())
            pi = self.get_pi(pi)
            pi = np.dot(pi[:,np.newaxis], pi[np.newaxis])
            return matrix*pi
            
        def sas_individual_pw(self, log_ratio=True, metric_type=None): return self.dis2met(-self.get_psi(log_ratio=log_ratio), metric_type)
        def sas_individual_out(self, log_ratio=True, metric_type=None): return self.sas_individual_pw(log_ratio, metric_type).sum(0)
        def sas_individual_in(self, log_ratio=True, metric_type=None): return self.sas_individual_pw(log_ratio, metric_type).sum(1)
        def sas_individual(self, log_ratio=True, metric_type=None):
            if self.directed: raise RuntimeError('individual SAS is ambiguous for directed SBMs; use sas_individual_out() or sas_individual_in()')
            else: return self.sas_individual_out(log_ratio, metric_type)
        def sas_community_pw(self, log_ratio=True, metric_type=None, pi=None): return self.ind2com(self.sas_individual_pw(log_ratio, metric_type), pi)
        def sas_community_out(self, log_ratio=True, metric_type=None, pi=None): return self.sas_community_pw(log_ratio, metric_type, pi).sum(0)
        def sas_community_in(self, log_ratio=True, metric_type=None, pi=None): return self.sas_community_pw(log_ratio, metric_type, pi).sum(1)
        def sas_community(self, log_ratio=True, metric_type=None, pi=None):
            if self.directed: raise RuntimeError('community SAS is ambiguous for directed SBMs; use sas_community_out() or sas_community_in()')
            else: return self.sas_community_out(log_ratio, metric_type, pi)
        def sas_global(self, log_ratio=True, metric_type=None, pi=None): return self.sas_community_pw(log_ratio, metric_type, pi).sum()
        
class NetworkData():

    def __init__(self, adjacency_matrix, membership_matrix=None, community_names=None, probability_matrix=None, data_name='network_data'):
        a, b = adjacency_matrix.shape
        if a!=b: raise ValueError('expected square adjacency matrix')
        self.name = data_name
        self.n = a
        self.adj = np.array(np.array(adjacency_matrix, dtype=bool), dtype=int)
        self.directed = (self.adj!=self.adj.transpose()).any()
        self.set_memberships(membership_matrix, community_names)
        self.set_probability(probability_matrix)
        
    def set_memberships(self, matrix=None, names=None):
        if matrix is None:
            self.k = 0
            self.mem = np.array()
            self.names = tuple()
        a, b = matrix.shape
        if a!=self.n: raise ValueError('expected number of rows to match number of nodes')
        matrix = np.array(np.array(matrix, dtype=bool), dtype=int)
        if not (all([i==1 for i in matrix.sum(1)]) and all([1 in i for i in matrix])): raise ValueError('expected single-memberships only')
        if names is None: names = tuple([str(i) for i in range(b)])
        elif len(names)==b: names = tuple([str(i) for i in names])
        else: raise ValueError('expected name for every community')
        self.k = b
        self.mem = matrix
        self.names = names
        
    def set_probability(self, matrix=None):
        if matrix is None: self.p = np.array()
        a, b = matrix.shape
        if a!=self.n or b!=self.n: raise ValueError('expected square probability matrix of size matching number of nodes')
        if not ((0<=matrix).all() and (matrix<=1).all()): raise ValueError('probabilities must lie between 0 and 1 (inclusive)')
        self.p = matrix.copy()
    
    def get_community(self): return np.nonzero(self.mem)[1]
        
    def sort(self, inplace=True):
        if self.mem is None: raise ValueError('no communities provided to sort nodes along')
        groups = self.get_community()
        idx = np.argsort(groups, kind='mergesort')
        if inplace:
            temp = self.adj[idx,:]
            self.adj = temp[:,idx]
            self.mem = self.mem[idx,:]
            if self.p is not None:
                temp = self.p[idx,:]
                self.p = temp[:,idx]
        else: return idx
        
    def degree_out(self): return self.adj.sum(0)
    def degree_in(self): return self.adj.sum(1)
    def degree(self):
        if self.directed: raise RuntimeError('degree is ambiguous for directed graphs; use degree_in() or degree_out()')
        else: return self.degree_out()
    def mean_degree_out(self): return self.degree_out().mean()
    def mean_degree_in(self): return self.degree_in().mean()
    def mean_degree(self): return self.degree_out().mean()
    
    def eigvals(self, real=True):
        if self.directed:
            if real: return sorted(np.real(np.linalg.eigvals(self.adj)), reverse=True)
            else: return sorted(np.linalg.eigvals(self.adj), reverse=True)
        else: return sorted(np.linalg.eigvalsh(self.adj), reverse=True)