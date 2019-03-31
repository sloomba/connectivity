from warnings import warn
from collections.abc import MutableSequence
import numpy as np

class EgocentricSBM(MutableSequence):
    
    def __init__(self, shape=tuple(), dims=None, name=None, filepath=None):
        if filepath is not None:
            self.__init__()
            self.load(filepath)
            if dims is not None: self.set_dims(dims)
            if name is not None: self.name = str(name)
            return
        if isinstance(shape, dict) or not self.iterable(shape): shape = (shape,)
        shape = tuple(shape)
        if shape and all([isinstance(b, dict) for b in shape]):            
            self.__init__()
            for b in shape: self.append(b)
            if dims is not None: self.set_dims(dims)
            if name is not None: self.name = str(name)
            return
        if shape and not all([isinstance(b, int) and b>0 for b in shape]): raise ValueError('number of blocks must be an integer greater than 0')
        self._precision = 8
        self._curridx = 0
        self.ndim = len(shape)
        self.shape = shape
        self.name = str(name)
        self.dims = tuple()
        self.dimsdict = dict()
        self.set_dims(dims)
        self.pi = tuple([(1/b,)*b for b in shape])
        self.rho = tuple([(1/b,)*b for b in shape])
        self.omega = tuple([(1.,)*b for b in shape])
        self.check_consistency()
        
    @property
    def precision(self): return self._precision
    
    @precision.setter
    def precision(self, value):
        if not isinstance(value, int) or value<0: raise ValueError('precision must be an integer no less than 0')
        if value<8: warn('setting a low "precision" can affect model consistency checks', RuntimeWarning)
        self._precision = value

    def get_params(self): return {'name':self.name, 'shape':self.shape, 'dims':self.dims, 'pi':self.pi, 'rho':self.rho, 'omega':self.omega, 'precision':self.precision}

    def set_params(self, params=dict()):
        params_old = self.get_params()
        if isinstance(params, dict): params = params.copy()
        else: params = dict(params)
        if 'name' not in params: params['name'] = params_old['name']
        if 'shape' not in params: params['shape'] = params_old['shape']
        if 'dims' not in params: params['dims'] = params_old['dims']
        if 'pi' not in params: params['pi'] = params_old['pi']
        if 'rho' not in params: params['rho'] = params_old['rho']
        if 'omega' not in params: params['omega'] = params_old['omega']
        if 'precision' not in params: params['precision'] = params_old['precision']
        try:
            self.__init__(params['shape'], params['dims'], params['name'])
            self.check_consistency(pi=params['pi'], rho=params['rho'], omega=params['omega'])
            self.pi = tuple([tuple(x) for x in params['pi']])
            self.rho = tuple([tuple(x) for x in params['rho']])
            self.omega = tuple([tuple(x) for x in params['omega']])
            self.precision = params['precision']
        except Exception as err:
            self.set_params(params_old)
            raise err

    def __str__(self):
        params = self.get_params()
        out = [['name:', params['name']]]
        out.append(['ndim:', str(self.ndim)])
        out.append(['shape:', str(params['shape'])])
        out.append(['', ''])
        out.append(['|dims|', '|blocks|'])
        out += [[str(x[0]), str(x[1])] for x in params['dims']]
        out.append(['', ''])
        out.append(['|dims|', '|pi|'])
        out += [[str(x[0]), str(y)] for x, y in zip(params['dims'], params['pi'])]
        out.append(['', ''])
        out.append(['|dims|', '|rho|'])
        out += [[str(x[0]), str(y)] for x, y in zip(params['dims'], params['rho'])]
        out.append(['', ''])
        out.append(['|dims|', '|omega|'])
        out += [[str(x[0]), str(y)] for x, y in zip(params['dims'], params['omega'])]
        width = max([len(i[0])+4 for i in out])
        for i in range(len(out)): out[i] = out[i][0].ljust(width) + out[i][1]
        return '\n'.join(out)
    
    def __len__(self): return self.ndim
    
    def __copy__(self):
        params = self.get_params()
        out = type(self)(params['shape'], params['dims'], params['name'])
        out.pi = params['pi']
        out.rho = params['rho']
        out.omega = params['omega']
        out.precision = params['precision']
        return out

    def copy(self): return self.__copy__()

    def save(self, filepath=None):
        if filepath is None: filepath = self.name
        if filepath[-4:]!='.ego': filepath += '.ego'
        import json
        json.dump(self.get_params(), open(filepath, 'w'))

    def load(self, filepath):
        import json
        self.set_params(json.load(open(filepath, 'r')))

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
            else: raise KeyError('sbm key "%s" not found; must be dimension name'%key)
        return key

    def keys(self): return tuple(self.dimsdict.keys())

    def values(self): return tuple([self[key] for key in self.keys()])

    def items(self): return tuple([(key, self[key]) for key in self.keys()])
    
    def __getitem__(self, key):
        if isinstance(key, slice): return [self[i] for i in range(*key.indices(len(self)))]
        key = self.get_key(key)
        return {'index':key,
                'name':self.dims[key][0], 
                'blocks':self.dims[key][1], 
                'pi':self.pi[key],
                'rho':self.rho[key], 
                'omega':self.omega[key]}

    def __setitem__(self, key, value):
        key = self.get_key(key)
        if not isinstance(value, dict): value = dict(value)
        params = dict()
        found = set()
        if 'pi' in value:
            params['pi'] = list(self.pi)
            params['pi'][key] = value['pi']
            found.add('pi')
        if 'rho' in value:
            params['rho'] = list(self.rho)
            params['rho'][key] = value['rho']
            found.add('rho')
        if 'omega' in value:
            params['omega'] = list(self.omega)
            params['omega'][key] = value['omega']
            found.add('omega')
        if 'name' in value or 'blocks' in value:
            params['dims'] = list(self.dims)
            params['dims'][key] = list(params['dims'][key])
            if 'name' in value:
                params['dims'][key][0] = value['name']
                found.add('name')
            if 'blocks' in value:
                params['dims'][key][1] = value['blocks']
                found.add('blocks')
        self.set_params(params)
        found = value.keys() - found
        if found: warn('no such settable attributes found: %s'%found)

    def insert(self, key, value):
        if not isinstance(value, dict): value = dict(value)
        if isinstance(key, int):
            if 'name' in value: keyname = value['name']
            else: keyname = None
        else:
            keyname = str(key)
            if 'index' in value: key = value['index']
            else: key = self.ndim
        params = self.get_params()
        for x in ['pi', 'rho', 'omega', 'dims', 'shape']: params[x] = list(params[x])
        shape = [len(value[x]) if self.iterable(value[x]) else -1 for x in ['pi', 'rho', 'omega', 'blocks'] if x in value]
        if 'num' in value: shape.append(value['num'])
        num = 0
        for i in range(len(shape)):
            if shape[i]>0:
                if not num: num = shape[i]
                else:
                    if num != shape[i]: raise ValueError('given attributes exhibit inconsistent number of blocks')
        if not num: raise ValueError('unable to infer number of blocks; set "num" attribute appropriately')
        if 'pi' in value: params['pi'].insert(key, value['pi'])
        else: params['pi'].insert(key, (1/num,)*num)
        if 'rho' in value: params['rho'].insert(key, value['rho'])
        else: params['rho'].insert(key, (1/num,)*num)
        if 'omega' in value: params['omega'].insert(key, value['omega'])
        else: params['omega'].insert(key, (self.mean_omega(),)*num)
        if 'blocks' in value: params['dims'].insert(key, (keyname, value['blocks']))
        else: params['dims'].insert(key, keyname)
        params['shape'].insert(key, num)
        self.set_params(params)
        found = value.keys() - {'pi', 'rho', 'omega', 'blocks', 'num', 'name', 'index'}
        if found: warn('no such settable attributes found: %s'%found)
            
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
        if isinstance(other, type(self)):
            x = self.copy()
            y = other.copy()
            x.sort()
            y.sort()
            return x.pi==y.pi and x.rho==y.rho and x.omega==y.omega
        else: return False
     
    def __lt__(self, other):
        if isinstance(other, type(self)):
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
        else: raise TypeError('can only compare two instances of type "EgocentricSBM"')
        
    def __le__(self, other):
        if self==other: return True
        else: return self<other

    def __iadd__(self, other):
        if isinstance(other, type(self)):
            for dim in other[:]: self.append(dim)
            return self
        else: raise TypeError('can only add two instances of type "EgocentricSBM"')

    def __add__(self, other):
        out = self.copy()
        out += other
        out.name = self.name+' + '+other.name
        out.precision = max(self.precision, other.precision)
        return out

    def kron(self, a, b=None, f=None):
        from itertools import product
        if b is None: b = a
        if f is None or f=='product': f = lambda x, y: x*y
        elif f=='concat': f = lambda x, y: '('+x+','+y+')'
        if all([self.iterable(i) for i in a]) and all([self.iterable(i) for i in b]): deep = True
        elif all([not self.iterable(i) for i in a]) and all([not self.iterable(i) for i in b]): deep = False
        else: raise ValueError('pass either iterables, or iterable of iterables')
        if deep: return tuple([tuple([f(i, j) for i, j in product(*x)]) for x in product(a, b)])
        else: return tuple([f(i, j) for i, j in product(a, b)])

    def scale(self, a=None, inplace=True):
        if a is None: a = 1/self.mean_omega()
        if not (isinstance(a, float) or isinstance(a, int)): raise TypeError('can only scale by "int" or "float"')
        if inplace: self.omega = tuple([tuple([o*a for o in om]) for om in self.omega])
        else:
            out = self.copy()
            out.omega = tuple([tuple([o*a for o in om]) for om in out.omega])
            return out

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int): return self.scale(other, inplace=False)
        elif isinstance(other, type(self)):
            dimsnew = tuple(zip(self.kron([x[0] for x in self.dims], [x[0] for x in other.dims], 'concat'), self.kron([x[1] for x in self.dims], [x[1] for x in other.dims], 'concat')))
            out = type(self)(self.kron(self.shape, other.shape), dimsnew, '('+self.name+' * '+other.name+')')
            out.pi = self.kron(self.pi, other.pi)
            out.rho = self.kron(self.rho, other.rho)
            out.omega = self.kron(self.omega, other.omega)
            out.precision = max(self.precision, other.precision)
            return out
        else: raise TypeError('multiplicand can be "int", "float" or "EgocentricSBM"')

    def __rmul__(self, other): return self*other

    def __pow__(self, other):
        if isinstance(other, int) and other>=0:
            if other==0: out = type(self)(1)
            else:
                out = self.copy()
                for i in range(1, other): out *= self
            out.name = '('+self.name+')**'+str(other)
            return out
        else: raise TypeError('can only exponentiate "EgocentricSBM" by a non-negative "int"')

    def __truediv__(self, other):
        if isinstance(other, float) or isinstance(other, int): return self.scale(1/other, inplace=False)
        else: raise TypeError('can only divide "EgocentricSBM" by "int" or "float"')

    def ishomophilous(self): return tuple([tuple([u<v for u,v in zip(i,j)]) for i,j in zip(self.pi, self.rho)])
    def isheterophilous(self): return tuple([tuple([u>v for u,v in zip(i,j)]) for i,j in zip(self.pi, self.rho)])
    def isambiphilous(self): return tuple([tuple([u==v for u,v in zip(i,j)]) for i,j in zip(self.pi, self.rho)])
    
    def iterable(self, obj):
        if isinstance(obj, str): return False
        try:
            iter(obj)
            return True
        except TypeError: return False
    
    def find_sum(self, list_of_vecs, approx=False, collapse=False):
        out = [sum(v) for v in list_of_vecs]
        if approx: out = [round(x, self.precision) for x in out]
        if collapse: out = sum(out)/len(out)
        return out

    def find_mean(self, list_of_vecs, pi=None, approx=False, collapse=False):
        if pi is None: pi = self.pi
        elif isinstance(pi, str) and pi=='uni': pi = tuple([(1/b,)*b for b in self.shape])
        out = [sum([i*j for i, j in zip(v, p)]) for v, p in zip(list_of_vecs, pi)]
        if approx: out = [round(x, self.precision) for x in out]
        if collapse: out = sum(out)/len(out)
        return out
    
    def sum_pi(self, pi=None, approx=False, collapse=False):
        if pi is None: pi = self.pi
        elif isinstance(pi, str) and pi=='uni': pi = tuple([(1/b,)*b for b in self.shape])
        return self.find_sum(pi, approx, collapse)
    
    def sum_rho(self, rho=None, approx=False, collapse=False):
        if rho is None: rho = self.rho
        return self.find_sum(rho, approx, collapse)
    
    def dev_pi(self, pi=None, approx=False, collapse=False):
        if pi is None: pi = self.pi
        elif isinstance(pi, str) and pi=='uni': pi = tuple([(1/b,)*b for b in self.shape])
        return self.find_mean(pi, pi, approx, collapse)
    
    def mean_rho(self, rho=None, pi=None, approx=False, collapse=False):
        if rho is None: rho = self.rho
        return self.find_mean(rho, pi, approx, collapse)
    
    def mean_omega(self, omega=None, pi=None, approx=False, collapse=True):
        if omega is None: omega = self.omega
        return self.find_mean(omega, pi, approx, collapse)
    
    def mean_homophily(self, rho=None, pi=None, approx=False, collapse=False): return self.mean_rho(rho, pi, approx, collapse)
    
    def mean_heterophily(self, rho=None, pi=None, approx=False, collapse=False):
        if collapse: 1 - self.mean_homophily(rho, pi, approx, collapse)
        else: return [1-x for x in self.mean_homophily(rho, pi, approx, collapse)]
    
    def mean_homoffinity(self, omega=None, rho=None, pi=None, approx=False, collapse=False):
        if rho is None: rho = self.rho
        if omega is None: omega = self.omega
        return self.find_mean([[w*m for w, m in zip(o, r)] for o, r in zip(omega, rho)], pi, approx, collapse)
    
    def mean_heteroffinity(self, omega=None, rho=None, pi=None, approx=False, collapse=False):
        if rho is None: rho = self.rho
        if omega is None: omega = self.omega
        return self.find_mean([[w*(1-m) for w, m in zip(o, r)] for o, r in zip(omega, rho)], pi, approx, collapse)

    def check_consistency(self, pi=None, omega=None, rho=None):
        if pi is None: pi = self.pi
        if omega is None: omega = self.omega
        if rho is None: rho = self.rho
        #pi tests
        if not self.iterable(pi): raise TypeError('pi must be an interable')
        if len(pi)!=self.ndim: raise ValueError('provide pi for every dimension')
        for i in range(len(pi)):
            if not self.iterable(pi[i]): raise TypeError('pi for dimension %s must be an interable'%i)
            if len(pi[i])!=self.shape[i]: raise ValueError('provide pi for all blocks of dimension %s'%i)
            if not all([0<j<=1 for j in pi[i]]): raise ValueError('pi must be between 0 (exclusive) and 1 (inclusive); see dimension %s'%i)
        if not all([p==1 for p in self.sum_pi(pi=pi, approx=True)]): raise ValueError('pi must sum up to 1')
        #omega tests
        if not self.iterable(omega): raise TypeError('omega must be an interable')
        if len(omega)!=self.ndim: raise ValueError('provide omega for every dimension')
        for i in range(len(omega)):
            if not self.iterable(omega[i]): raise TypeError('omega for dimension %s must be an interable'%i)
            if len(omega[i])!=self.shape[i]: raise ValueError('provide omega for all blocks of dimension %s'%i)
            if not all([0<=j for j in omega[i]]): raise ValueError('omega must be >=0; see dimension %s'%i)
        mean_omega = self.mean_omega(omega=omega, pi=pi, approx=True, collapse=False)
        for i in range(1, len(mean_omega)):
            if not mean_omega[i] == mean_omega[0]: raise ValueError('mean omega must remain same')
        #rho tests
        if not self.iterable(rho): raise TypeError('rho must be an interable')
        if len(rho)!=self.ndim: raise ValueError('provide rho for every dimension')
        for i in range(len(rho)):
            if not self.iterable(rho[i]): raise TypeError('rho for dimension %s must be an interable'%i)
            if len(rho[i])!=self.shape[i]: raise ValueError('provide rho for all blocks of dimension %s'%i)
            if not all([0<=j<=1 for j in rho[i]]): raise ValueError('rho must be between 0 and 1 (inclusive); see dimension %s'%i)

    def get_dims(self):
        from itertools import product
        dims_flat = [[(dim[0]+':'+str(d)) for d in dim[1]] for dim in self.dims]
        return tuple([','.join(i) for i in product(*dims_flat)])
    
    def get_nextdim(self, dims=None, prefix=''):
        from itertools import count
        for i in range(len(prefix)):
            if prefix[i:].isdigit():
                prefix = prefix[:i]
                break
        if dims is None: dims = self.dimsdict
        dimnum = [int(d[len(prefix):]) for d in dims if d is not None and d[:len(prefix)]==prefix and d[len(prefix):].isdigit()]
        if dimnum: return map(lambda x: prefix+str(x), count(max(dimnum)+1))
        else: return map(lambda x: prefix+str(x), count())
    
    def set_dims(self, dims=None):
        if dims is None: dims = [None for i in range(self.ndim)]
        if not self.iterable(dims): raise ValueError('provide names for each dimension')
        dims = list(dims)
        if len(dims)!=self.ndim: raise ValueError('provide names for each dimension')
        for i in range(self.ndim):
            if self.iterable(dims[i]):
                found = False
                if len(dims[i])==2:
                    if (not self.iterable(dims[i][0])) and self.iterable(dims[i][1]):
                        if len(dims[i][1])==self.shape[i]:
                            found = True
                            dims[i] = [dims[i][0], tuple(dims[i][1])]
                        else: raise ValueError('provide names for each block of dimension %d'%i)
                if (not found) and len(dims[i])==self.shape[i]:
                    found = True
                    dims[i] = [None, tuple(dims[i])]
                if not found: raise ValueError('provide names for each block of dimension %d'%i)
            else: dims[i] = [dims[i], tuple([str(j) for j in range(self.shape[i])])]
        dimcount = {None:self.get_nextdim([x[0] for x in dims])}
        dimset = []
        for i in range(self.ndim):
            if dims[i][0] is None: dims[i][0] = next(dimcount[dims[i][0]])
            if not isinstance(dims[i][0], str): dims[i][0] = str(dims[i][0])
            if dims[i][0] in dimset:
                if dims[i][0] not in dimcount: dimcount[dims[i][0]] = self.get_nextdim([x[0] for x in dims], dims[i][0])
                dims[i][0] = next(dimcount[dims[i][0]])
            dimset.append(dims[i][0])
            dims[i] = tuple(dims[i]) 
        self.dims = tuple(dims)
        self.dimsdict = dict(zip(dimset, range(self.ndim)))
        
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
            if omega[pair[0]]!=omega[pair[1]]: temp1 = ((self.mean_omega() - sum([o*q for (o, q) in zip(omega, p)])) - omega[pair[0]]*t)/(omega[pair[1]]-omega[pair[0]])
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
    
    def get_model(self, mode='full', directed=True, name='sbm'): return self.StochasticBlockModel(self, mode=mode, directed=directed, name=name)
    
    class StochasticBlockModel():
        
        def __init__(self, ego=None, mode='full', directed=True, name=None, filepath=None):
            
            if isinstance(ego, str):
                filepath = ego
                ego = EgocentricSBM()
                ego.load(filepath)
            elif ego is None:
                if filepath is None: raise ValueError('either provide "ego" as a valid EgocentricSBM, or as the path to an .ego file, or provide "filepath" to an .npz file containing the model ')
                self.load(filepath)
                if name is not None: self.name = str(name)
                return
            if mode not in ['full', 'pp', 'ppcollapsed']: raise ValueError('invalid model mode "%s"'%mode)
            self.mode = mode
            self.name = str(name)
            self.ndim = ego.ndim
            self.shape = ego.shape
            self.dims = ego.get_dims()
            self.pi = ego.pi
            self.meanomega = ego.mean_omega()
            self._precision = ego.precision
            
            if mode=='full':
                self.params = tuple([tuple([(o, r/p, (1-r)/(1-p)) if p!=1 else (o, 1., 0.) for r, o, p in zip(ego.rho[i], ego.omega[i], ego.pi[i])]) for i in range(ego.ndim)])
                self.psi = tuple([(np.diag([x[0] for x in item]), np.diag([x[1]-x[2] for x in item])+np.vstack([x[2]*np.ones(len(item)) for x in item])) for item in self.params])
            elif mode=='pp': #planted-partition model
                self.params = (self.meanomega, tuple([(r/p, (1-r)/(1-p)) if p!=1 else (1., 0.) for r, p in zip(ego.mean_rho(), ego.dev_pi())]))
                self.psi = tuple([(self.params[0]*np.eye(ego.shape[i]), (self.params[1][i][0]-self.params[1][i][1])*np.eye(ego.shape[i])+self.params[1][i][1]*np.ones([ego.shape[i], ego.shape[i]])) for i in range(ego.ndim)])
            elif mode=='ppcollapsed': #planted-partition model collapsed
                self.params = tuple([(h/d, (self.meanomega-h)/(1-d)) if d!=1 else (self.meanomega, 0.) for h, d in zip(ego.mean_homoffinity(), ego.dev_pi())])
                self.psi = tuple([(self.params[i][0]-self.params[i][1])*np.eye(ego.shape[i])+self.params[i][1]*np.ones([ego.shape[i], ego.shape[i]]) for i in range(ego.ndim)])
            
            temp = self.get_psi(directed=True)
            self.directed = directed and (temp!=temp.transpose()).any()
                
        @property
        def precision(self): return self._precision

        @precision.setter
        def precision(self, value):
            if not isinstance(value, int) or value<0: raise ValueError('precision must be an integer no less than 0')
            if value<8: warn('setting a low "precision" can affect model consistency checks', RuntimeWarning)
            self._precision = value
            
        def __str__(self):
            out = [['name:', self.name]]
            out.append(['mode:', self.mode])
            out.append(['ndim:', str(self.ndim)])
            out.append(['shape:', str(self.shape)])
            out.append(['pi:', str(self.pi)])
            out.append(['params:', str(self.params)])
            width = max([len(i[0])+4 for i in out])
            for i in range(len(out)): out[i] = out[i][0].ljust(width) + out[i][1]
            return '\n'.join(out)
    
        def __len__(self): return self.get_shape()

        def __eq__(self, other):
            if isinstance(other, type(self)): return (self.get_psi(sort=True)==other.get_psi(sort=True)).all()
            else: return False

        def save(self, filepath=None):
            if filepath is None: filepath = self.name
            np.savez(filepath, name=self.name, mode=self.mode, ndim=self.ndim, shape=self.shape, dims=self.dims, params=self.params, pi=self.pi, psi=self.psi, meanomega=self.meanomega, precision=self.precision, directed=self.directed)

        def load(self, filepath):
            file = np.load(filepath)
            self.name = str(file['name'])
            self.mode = str(file['mode'])
            self.ndim = int(file['ndim'])
            self.shape = tuple(file['shape'])
            self.dims = tuple(file['dims'])
            self.params = tuple(file['params'])
            self.pi = tuple(file['pi'])
            self.psi = tuple(file['psi'])
            self.meanomega = float(file['meanomega'])
            self.precision = int(file['precision'])
            self.directed = bool(file['directed'])
            
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
            
        def get_psi(self, directed=None, log_ratio=False, sort=False):
            multiplier = 1/(self.meanomega**(self.ndim-1))
            if self.mode=='ppcollapsed': out = multiplier*self.kron(self.psi)
            elif self.mode=='pp': out = multiplier*np.matmul(self.kron([x[0] for x in self.psi]), self.kron([x[1] for x in self.psi]))
            elif self.mode=='full': out =  multiplier*np.matmul(self.kron([x[0] for x in self.psi]), self.kron([x[1] for x in self.psi]))
            if directed is None: directed = self.directed
            if not directed:
                if self.directed: warn('symmetrising the stochastic block matrix', RuntimeWarning)
                out = (out + out.transpose())/2
            if sort:
                idx = np.argsort(np.diag(out), kind='mergesort')[::-1] #sort by decreasing homophily
                out = out[idx,:][:,idx]
            if log_ratio: out = np.log2(out) - np.hstack([np.log2(np.diag(out))[:,np.newaxis]]*self.get_shape())
            return out

        def find_mean(self, matrix, pi=None): return np.dot(matrix, self.get_pi(pi))

        def generate_people(self, n=100, pi=None): return np.random.multinomial(1, self.get_pi(pi), n)
        
        def generate_network(self, people, directed=None):
            n, d = people.shape
            people = np.array(np.array(people, dtype=bool), dtype=int)
            factor = 1/(n-1) #correction to keep graph simple
            if d!=self.get_shape(): raise ValueError('number of columns must correspond to full number of blocks')
            if not (all([i==1 for i in people.sum(1)]) and all([1 in i for i in people])): raise ValueError('expected single-memberships only')
            keep = people.sum(axis=0)>0 #ignoring communities with 0 members
            people = people[:,keep]
            psi = self.get_psi()[keep,:][:,keep]
            p = np.matmul(np.matmul(people, factor*psi), people.transpose())
            p[p>1] = 1.
            np.fill_diagonal(p, 0)
            a = np.random.binomial(1, p)
            if directed is None: directed = self.directed
            if not directed:
                if self.directed: warn('generating an undirected network from a directed SBM', RuntimeWarning)
                a = np.triu(a)
                a = a + a.transpose()
            return (a, p tuple(np.array(self.dims)[keep]))
        
        def generate_networkdata(self, n=100, pi=None, directed=False, name='network_data'):
            z = self.generate_people(n, pi)
            a, p, d = self.generate_network(z, directed)
            return NetworkData(a, z, d, p, name)

        def generate_networkx(self, n=100, pi=None, directed=False, selfloops=False):
            if selfloops: factor = 1/n
            else: factor = 1/(n-1)
            people = self.generate_people(n, pi).sum(axis=0)
            keep = people>0
            people = people[keep]
            p = factor*(self.get_psi(directed)[keep,:][:,keep])
            p[p>1] = 1
            params = dict(sizes=people.tolist(), p=p.tolist(), directed=directed, selfloops=selfloops)
            try:
                from networkx import stochastic_block_model 
                return stochastic_block_model(**params)
            except:
                return params
        
        def eigvals_pipsi(self, pi=None, directed=None, real=True):
            pi = self.get_pi(pi)
            if real: return sorted(np.real(np.linalg.eigvals(np.matmul(np.diag(pi), self.get_psi(directed)))), reverse=True)
            else: return sorted(np.linalg.eigvals(np.matmul(np.diag(pi), self.get_psi(directed))), reverse=True)
        
        def eigvals_pipsi_theoretical(self):
            pi = 1/self.get_shape()
            if self.mode=='full': raise RuntimeError('cannot compute theoretical eigenvalues for a "full" model')
            elif self.mode=='pp':
                factor = pi*self.meanomega
                eigs = sorted(tuple(factor*self.kron([(h[0]+h[1]*(s-1),)+(h[0]-h[1],)*(s-1) for (h, s) in zip(self.params[1], self.shape)])), reverse=True)
            elif self.mode=='ppcollapsed':
                factor = pi/(self.meanomega**(self.ndim-1))
                eigs = sorted(tuple(factor*self.kron([(h[0]+h[1]*(s-1),)+(h[0]-h[1],)*(s-1) for (h, s) in zip(self.params, self.shape)])), reverse=True)
            return eigs
        
        def homoffinity_out(self, pi=None, directed=None, log_ratio=False): return np.diag(self.get_psi(directed, log_ratio))*self.get_pi(pi)
        def homoffinity_in(self, pi=None, directed=None, log_ratio=False): return self.homoffinity_out(pi, directed, log_ratio)
        def homoffinity(self, pi=None, directed=None, log_ratio=False): return self.homoffinity_out(pi, directed, log_ratio)
            
        def heteroffinity_out(self, pi=None, directed=None, log_ratio=False): return self.affinity_out(pi, directed, log_ratio)-self.homoffinity_out(pi, directed, log_ratio)
        def heteroffinity_in(self, pi=None, directed=None, log_ratio=False): return self.affinity_in(pi, directed, log_ratio)-self.homoffinity_in(pi, directed, log_ratio)
        def heteroffinity(self, pi=None, directed=None, log_ratio=False): return self.affinity(pi, directed, log_ratio)-self.homoffinity(pi, directed, log_ratio)
        
        def affinity_out(self, pi=None, directed=None, log_ratio=False): return self.find_mean(self.get_psi(directed, log_ratio), pi)
        def affinity_in(self, pi=None, directed=None, log_ratio=False): return self.find_mean(self.get_psi(directed, log_ratio).transpose(), pi)
        def affinity(self, pi=None, directed=None, log_ratio=False):
            if self.directed and (directed or directed is None): raise RuntimeError('affinity is ambiguous for directed SBMs; use affinity_in() or affinity_out()')
            else: return self.affinity_out(pi, directed, log_ratio)
        
        def mean_homoffinity_out(self, pi=None, directed=None, log_ratio=False): return self.find_mean(self.homoffinity_out(pi, directed, log_ratio), pi)
        def mean_homoffinity_in(self, pi=None, directed=None, log_ratio=False): return self.find_mean(self.homoffinity_in(pi, directed, log_ratio), pi)
        def mean_homoffinity(self, pi=None, directed=None, log_ratio=False): return self.find_mean(self.homoffinity(pi, directed, log_ratio), pi)
        
        def mean_heteroffinity_out(self, pi=None, directed=None, log_ratio=False): return self.find_mean(self.heteroffinity_out(pi, directed, log_ratio), pi)
        def mean_heteroffinity_in(self, pi=None, directed=None, log_ratio=False): return self.find_mean(self.heteroffinity_in(pi, directed, log_ratio), pi)
        def mean_heteroffinity(self, pi=None, directed=None, log_ratio=False): return self.find_mean(self.heteroffinity_out(pi, directed, log_ratio), pi)
        
        def mean_affinity_out(self, pi=None, directed=None, log_ratio=False): return self.find_mean(self.affinity_out(pi, directed, log_ratio), pi)
        def mean_affinity_in(self, pi=None, directed=None, log_ratio=False): return self.find_mean(self.affinity_in(pi, directed, log_ratio), pi)
        def mean_affinity(self, pi=None, directed=None, log_ratio=False): return self.find_mean(self.affinity_out(pi, directed, log_ratio), pi)
        
        def floyd_warshall(self, pw_dist_matrix, paths=True):
            if (pw_dist_matrix<0).any(): warn('negative pairwise distances can lead to negative path lengths')
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
                else: raise ValueError ('metric type "%s" not found'%metric_type)
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
        
        def sas_individual_pw(self, log_ratio=True, metric_type=None): return self.dis2met(-self.get_psi(log_ratio=log_ratio), metric_type)
        def sas_individual_out(self, log_ratio=True, metric_type=None, pi=None): return self.find_mean(self.sas_individual_pw(log_ratio, metric_type), pi)
        def sas_individual_in(self, log_ratio=True, metric_type=None, pi=None): return self.find_mean(self.sas_individual_pw(log_ratio, metric_type).transpose(), pi)
        def sas_individual(self, log_ratio=True, metric_type=None, pi=None):
            if self.directed: raise RuntimeError('individual SAS is ambiguous for directed SBMs; use sas_individual_out() or sas_individual_in()')
            else: return self.sas_individual_out(log_ratio, metric_type, pi)
        def sas_global(self, log_ratio=True, metric_type=None, pi=None): return self.find_mean(self.sas_individual_out(log_ratio, metric_type, pi), pi)
        
class NetworkData():

    def __init__(self, adjacency_matrix=None, membership_matrix=None, community_names=None, probability_matrix=None, name=None, filepath=None):
        if adjacency_matrix is None:
            if filepath is None: raise ValueError('either provide "adjacency_matrix" as a valid square numpy array, or provide "filepath" to an .npz file containing the data')
            self.load(filepath)
            if name is not None: self.name = str(name)
            return
        a, b = adjacency_matrix.shape
        if a!=b: raise ValueError('expected square adjacency matrix')
        self.name = str(name)
        self.n = a
        self.adj = np.array(np.array(adjacency_matrix, dtype=bool), dtype=int)
        self.directed = (self.adj!=self.adj.transpose()).any()
        self.set_memberships(membership_matrix, community_names)
        self.set_probability(probability_matrix)

    def __str__(self):
        out = [['name:', self.name]]
        out.append(['n:', str(self.n)])
        out.append(['k:', str(self.k)])
        out.append(['directed', str(self.directed)])
        out.append(['mean degree:', str(self.mean_degree())])
        width = max([len(i[0])+4 for i in out])
        for i in range(len(out)): out[i] = out[i][0].ljust(width) + out[i][1]
        return '\n'.join(out)

    def __len__(self): return self.n

    def save(self, filepath=None):
        if filepath is None: filepath = self.name
        np.savez(filepath, name=self.name, names=self.names, n=self.n, k=self.k, directed=self.directed, adj=self.adj, mem=self.mem, p=self.p)

    def load(self, filepath):
        file = np.load(filepath)
        self.name = str(file['name'])
        self.names = tuple(file['names'])
        self.n = int(file['n'])
        self.k = int(file['k'])
        self.directed = bool(file['directed'])
        self.adj = file['adj']
        self.mem = file['mem']
        self.p = file['p']

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