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
        import json
        try: json.dump(self.get_params(), filepath)
        except Exception as err:
            if filepath is None: filepath = self.name
            if filepath[-4:]!='.ego': filepath += '.ego'        
            json.dump(self.get_params(), open(filepath, 'w'))

    def load(self, filepath):
        import json
        try: self.set_params(json.load(filepath))
        except Exception as err: self.set_params(json.load(open(filepath, 'r')))

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
    
    def get_key(self, keys=None):
        if keys is None: return tuple(range(self.ndim))
        if not self.iterable(keys): keys = [keys]
        key_index = list()
        for key in keys:
            if key in self.dimsdict:
                key_index.append(self.dimsdict[key])
            else:
                if isinstance(key, int):
                    if key<0 and key>=-self.ndim: key += self.ndim
                    if key>=0 and key<self.ndim: pass
                    else: raise IndexError('sbm index out of range; must be less than ndims')
                else: raise KeyError('sbm key "%s" not found; must be dimension name'%key)
                key_index.append(key)
        if len(key_index)==1: return key_index[0]
        else: return tuple(key_index)

    def get_block(self, key, blocks=None):
        key = self.get_key(key)
        if self.iterable(key): raise ValueError('expected a single key to retrieve blocks for')
        if blocks is None: return tuple(range(self.shape[key]))
        if not self.iterable(blocks): blocks = [blocks]
        block_names = dict(zip(self.dims[key][1], range(self.shape[key])))
        if len(block_names)!=self.shape[key]: warn('dimension %d has duplicate block names; last index of blocks matched to same name is returned'%key, RuntimeWarning)
        block_index = list()
        for blk in blocks:
            if blk in block_names: block_index.append(block_names[blk])
            else:
                if isinstance(blk, int):
                    if blk<0 and blk>=-self.shape[key]: blk += self.shape[key]
                    if blk>=0 and blk<self.shape[key]: pass
                    else: raise IndexError('dim index out of range; must be less than size of dimension %d'%key)
                else: raise KeyError('dim key "%s" not found; must be block name of dimension %d'%(blk, key))
                block_index.append(blk)
        if len(block_index)==1: return block_index[0]
        else: return tuple(block_index)

    def get_name(self, key):
        key = self.get_key(key)
        if self.iterable(key): return tuple([self.dims[k][1] for k in key])
        else: return self.dims[key][1]

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
        
    def apx(self, param):
        if self.iterable(param): return [self.apx(p) for p in param]
        else:
            if isinstance(param, float) or isinstance(param, int): return round(param, self.precision)
            else: raise ValueError('expect int or float but got %s'%str(type(param)))

    def __eq__(self, other):
        if isinstance(other, type(self)):
            x = self.copy()
            y = other.copy()
            x.sort()
            y.sort()
            return self.apx(x.pi)==self.apx(y.pi) and self.apx(x.rho)==self.apx(y.rho) and self.apx(x.omega)==self.apx(y.omega)
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
                x_pi = self.apx(x.pi)
                y_pi = self.apx(y.pi)
                x_rho = self.apx(x.rho)
                y_rho = self.apx(y.rho)
                x_omega = self.apx(x.omega)
                y_omega = self.apx(y.omega)
                while i<self.ndim:
                    found = False
                    while j<other.ndim:
                        if x_pi[i]==y_pi[j] and x_rho[i]==y_rho[j] and x_omega[i]==y_omega[j]:
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

    def flatten(self, inplace=True):
        if len(self)>1:
            out = type(self)(self[0])
            for i in range(1, len(self)):
                out *= type(self)(self[i])
            out.scale(1/(self.mean_omega()**(len(self)-1)))
            out.name = '/'+self.name+'\\'
            if inplace: self.set_params(out.get_params())
            else: return out
        elif not inplace: return self.copy()
    
    def iterable(self, obj):
        if isinstance(obj, str): return False
        try:
            iter(obj)
            return True
        except TypeError: return False
    
    def find_sum(self, list_of_vecs, approx=False, collapse=False):
        out = [sum(v) for v in list_of_vecs]
        if collapse: out = sum(out)/len(out)
        if approx: out = self.apx(out)
        return out

    def find_ratio(self, list_of_vals_1, list_of_vals_2, log=False, approx=False, collapse=False):
        def try_log(x):
            from math import log2
            try: return log2(x)
            except: return -float('inf')
        if log: out = [try_log(i)-try_log(j) for i, j in zip(list_of_vals_1, list_of_vals_2)]
        else: out = [i/j for i, j in zip(list_of_vals_1, list_of_vals_2)]
        if collapse: out = sum(out)/len(out)
        if approx: out = self.apx(out)
        return out

    def find_mean(self, list_of_vecs, pi=None, approx=False, collapse=False):
        if pi is None: pi = self.pi
        elif isinstance(pi, str) and pi=='uni': pi = tuple([(1/b,)*b for b in self.shape])
        out = [sum([i*j for i, j in zip(v, p)]) for v, p in zip(list_of_vecs, pi)]
        if collapse: out = sum(out)/len(out)
        if approx: out = self.apx(out)
        return out

    def find_var(self, list_of_vecs, pi=None, approx=False, collapse=False, mean=None):
        if mean is None: mean = self.find_mean(list_of_vecs, pi)
        var = self.find_mean([[(i-m)**2 for i in v] for v, m in zip(list_of_vecs, mean)], pi, approx, collapse)
        return var
    
    def ishomophilous(self): return [[u<v for u,v in zip(i,j)] for i,j in zip(self.apx(self.pi), self.apx(self.rho))]
    
    def isheterophilous(self): return [[u>v for u,v in zip(i,j)] for i,j in zip(self.apx(self.pi), self.apx(self.rho))]
    
    def isambiphilous(self): return [[u==v for u,v in zip(i,j)] for i,j in zip(self.apx(self.pi), self.apx(self.rho))]

    def homophily(self, log=True, approx=False): return [self.find_ratio(j, i, log, approx) for i,j in zip(self.pi, self.rho)]
    
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

    def var_rho(self, rho=None, pi=None, approx=False, collapse=False):
        if rho is None: rho = self.rho
        return self.find_var(rho, pi, approx, collapse)
    
    def mean_omega(self, omega=None, pi=None, approx=False, collapse=True):
        if omega is None: omega = self.omega
        return self.find_mean(omega, pi, approx, collapse)

    def var_omega(self, omega=None, pi=None, approx=False, collapse=True):
        if omega is None: omega = self.omega
        return self.find_var(omega, pi, approx, collapse)

    def correct_pi(self, pi): return [[i/sum(p) for i in p] for p in pi]

    def correct_omega(self, omega, pi=None):
        omega = list(omega)
        mean_omega = self.mean_omega(omega, pi, collapse=False)
        mo = sum(mean_omega)/len(mean_omega)
        for i in range(len(omega)):
            omega[i] = list(omega[i])
            for j in range(len(omega[i])):
                omega[i][j] *= mo/mean_omega[i]
        return omega

    def correct_rho(self, rho, omega=None):
        rho = list(rho)
        if omega is None: omega = self.omega
        for i in range(len(rho)):
            rho[i] = list(rho[i])
            for j in range(len(rho[i])):
                if omega[i][j]==0.: rho[i][j]=0.
        return rho

    def correct_params(self, pi, omega, rho):
        pi = self.correct_pi(pi)
        omega = self.correct_omega(omega, pi)
        rho = self.correct_rho(rho, omega)
        return pi, omega, rho
    
    def mean_homophily(self, rho=None, pi=None, approx=False, collapse=False): return self.mean_rho(rho, pi, approx, collapse)

    def var_homophily(self, rho=None, pi=None, approx=False, collapse=False): return self.var_rho(rho, pi, approx, collapse)

    def relative_mean_homophily(self, rho=None, pi=None, log=True, approx=False, collapse=False): return self.find_ratio(self.mean_homophily(rho, pi), self.dev_pi(pi), log, approx, collapse)
    
    def mean_heterophily(self, rho=None, pi=None, approx=False, collapse=False):
        if collapse: 1 - self.mean_homophily(rho, pi, approx, collapse)
        else: return [1-x for x in self.mean_homophily(rho, pi, approx, collapse)]

    def var_heterophily(self, rho=None, pi=None, approx=False, collapse=False): return self.var_homophily(rho, pi, approx, collapse)

    def relative_mean_heterophily(self, rho=None, pi=None, log=True, approx=False, collapse=False): return self.find_ratio(self.mean_heterophily(rho, pi), [1-x for x in self.dev_pi(pi)], log, approx, collapse)
    
    def mean_homoffinity(self, omega=None, rho=None, pi=None, approx=False, collapse=False):
        if rho is None: rho = self.rho
        if omega is None: omega = self.omega
        return self.find_mean([[w*m for w, m in zip(o, r)] for o, r in zip(omega, rho)], pi, approx, collapse)

    def var_homoffinity(self, omega=None, rho=None, pi=None, approx=False, collapse=False):
        if rho is None: rho = self.rho
        if omega is None: omega = self.omega
        return self.find_var([[w*m for w, m in zip(o, r)] for o, r in zip(omega, rho)], pi, approx, collapse)
    
    def mean_heteroffinity(self, omega=None, rho=None, pi=None, approx=False, collapse=False):
        if rho is None: rho = self.rho
        if omega is None: omega = self.omega
        return self.find_mean([[w*(1-m) for w, m in zip(o, r)] for o, r in zip(omega, rho)], pi, approx, collapse)

    def var_heteroffinity(self, omega=None, rho=None, pi=None, approx=False, collapse=False):
        if rho is None: rho = self.rho
        if omega is None: omega = self.omega
        return self.find_var([[w*(1-m) for w, m in zip(o, r)] for o, r in zip(omega, rho)], pi, approx, collapse)

    def check_consistency(self, pi=None, omega=None, rho=None):
        if pi is None: pi = self.pi
        if omega is None: omega = self.omega
        if rho is None: rho = self.rho
        #pi tests
        if not self.iterable(pi): raise TypeError('pi must be an iterable')
        if len(pi)!=self.ndim: raise ValueError('provide pi for every dimension')
        for i in range(len(pi)):
            if not self.iterable(pi[i]): raise TypeError('pi for dimension %s must be an iterable'%i)
            if len(pi[i])!=self.shape[i]: raise ValueError('provide pi for all blocks of dimension %s'%i)
            if not all([0<j<=1 for j in pi[i]]): raise ValueError('pi must be between 0 (exclusive) and 1 (inclusive); see dimension %s'%i)
        if not all([p==1 for p in self.sum_pi(pi=pi, approx=True)]): raise ValueError('pi must sum up to 1')
        #omega tests
        if not self.iterable(omega): raise TypeError('omega must be an iterable')
        if len(omega)!=self.ndim: raise ValueError('provide omega for every dimension')
        for i in range(len(omega)):
            if not self.iterable(omega[i]): raise TypeError('omega for dimension %s must be an iterable'%i)
            if len(omega[i])!=self.shape[i]: raise ValueError('provide omega for all blocks of dimension %s'%i)
            if not all([0<=j for j in omega[i]]): raise ValueError('omega must be >=0; see dimension %s'%i)
        mean_omega = self.mean_omega(omega=omega, pi=pi, approx=True, collapse=False)
        for i in range(1, len(mean_omega)):
            if not mean_omega[i] == mean_omega[0]: raise ValueError('mean omega must remain same')
        #rho tests
        if not self.iterable(rho): raise TypeError('rho must be an iterable')
        if len(rho)!=self.ndim: raise ValueError('provide rho for every dimension')
        for i in range(len(rho)):
            if not self.iterable(rho[i]): raise TypeError('rho for dimension %s must be an iterable'%i)
            if len(rho[i])!=self.shape[i]: raise ValueError('provide rho for all blocks of dimension %s'%i)
            if not all([0<=j<=1 for j in rho[i]]): raise ValueError('rho must be between 0 and 1 (inclusive); see dimension %s'%i)
            if not all(rho[i][j]==0 for j in range(self.shape[i]) if omega[i][j]==0): raise ValueError('rho must be 0 for a community with omega 0; see dimension %s'%i)

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

    def merge(self, key, blocks, name=None, inplace=True):
        key = self.get_key(key)
        blocks = self.get_block(key, blocks)
        if not blocks or not self.iterable(blocks):
            warn('expected at least 2 blocks to be merged together, no merge performed for dim %s'%str(key), RuntimeWarning)
            return
        shape = list(self.shape)
        shape[key] += 1-len(blocks) 
        dims = list(self.dims)
        dims[key] = list(dims[key])
        dims[key][1] = list(dims[key][1])
        dims_new = list()
        pi = list(self.pi)
        pi[key] = list(pi[key])
        pi_new = list()
        rho = list(self.rho)
        rho[key] = list(rho[key])
        rho_new = list()
        omega = list(self.omega)
        omega[key] = list(omega[key])
        omega_new = list()
        new_idx = blocks[0]
        blocks = sorted(blocks)
        for i in range(len(blocks)):
            dims_new.append(dims[key][1].pop(blocks[i]-i))
            pi_new.append(pi[key].pop(blocks[i]-i))
            rho_new.append(rho[key].pop(blocks[i]-i))
            omega_new.append(omega[key].pop(blocks[i]-i))
        sum_pi = sum(pi_new)
        sum_om = sum([p*o for p, o in zip(pi_new, omega_new)])
        if name is None: name = ' | '.join(['('+str(b)+')' for b in dims_new])
        dims[key][1].insert(new_idx, name)
        pi[key].insert(new_idx, sum_pi)
        omega[key].insert(new_idx, sum_om/sum_pi)
        if sum_om == 0: rho[key].insert(new_idx, 0.)
        else: rho[key].insert(new_idx, sum([p*o*r + p*o*(sum_pi-p)*(1-r)/(1-p) for p, o, r in zip(pi_new, omega_new, rho_new)])/sum_om)
        if inplace: self.set_params({'shape':shape, 'dims':dims, 'pi':pi, 'rho':rho, 'omega':omega})
        else:
            x = self.copy()
            x.set_params({'shape':shape, 'dims':dims, 'pi':pi, 'rho':rho, 'omega':omega})
            return x

    def merge_inverse(self, key, block, pi_split=None, omega_split=None, rho_split=None, names=None, inplace=True):
        key = self.get_key(key)
        block = self.get_block(key, block)
        if self.iterable(block): raise TypeError('expected exactly 1 block to be split at a time')
        if pi_split is None: pi_split = 1.0
        if not self.iterable(pi_split):
            num_splits = round(1/pi_split)
            pi_split = [1/num_splits]*num_splits
        else: num_splits = len(pi_split)
        if not all([0<i<=1 for i in pi_split]): raise ValueError('pi_split (splitting proportions) must be between 0 (exclusive) and 1 (inclusive)')
        if self.sum_pi(pi=[pi_split], approx=True)[0]!=1: raise ValueError('pi_split (splitting proportions) must sum up to 1')
        if num_splits==1: return
        shape = list(self.shape)
        shape[key] += num_splits-1
        dims = list(self.dims)
        dims[key] = list(dims[key])
        dims[key][1] = list(dims[key][1])
        dims_old = str(dims[key][1].pop(block))
        if names is None: names = dims_old
        if not self.iterable(names): names = [names+' !| '+str(i+1) for i in range(num_splits)]
        if len(names)!=num_splits: raise TypeError('expected names for exactly %d new blocks'%num_splits)
        pi = list(self.pi)
        pi[key] = list(pi[key])
        pi_old = pi[key].pop(block)
        rho = list(self.rho)
        rho[key] = list(rho[key])
        rho_old = rho[key].pop(block)
        omega = list(self.omega)
        omega[key] = list(omega[key])
        omega_old = omega[key].pop(block)
        if omega_split is None: omega_split = omega_old
        if not self.iterable(omega_split): omega_split = [omega_split]*num_splits
        if len(omega_split)!=num_splits: raise ValueError('expected omega_split for exactly %d blocks'%num_splits)
        if not all([0<=i for i in omega_split]): raise ValueError('omega_split must be >=0')
        if self.mean_omega(omega=[omega_split], pi=[pi_split], approx=True, collapse=False)[0]!=self.apx(omega_old): raise ValueError('given omega_split does not satisfy mean omega constraint; retry without omega_split or with an apt one')
        constant_1 = sum([o*p*(1-p)/(1-pi_old*p) for p, o in zip(pi_split, omega_split)])
        constant_2 = sum([p**2/(1-pi_old*p) for p in pi_split])
        coefficient = (omega_old*rho_old - pi_old*constant_1)/(constant_2*(1-pi_old))
        if rho_split is None: rho_split = [coefficient*p/o for p, o in zip(pi_split, omega_split)]
        if not self.iterable(rho_split): rho_split = [rho_split]*num_splits
        if len(rho_split)!=num_splits: raise ValueError('expected rho_split for exactly %d blocks'%num_splits)
        if not all([0<=i<=1 for i in rho_split]): raise ValueError('rho_split must be between 0 and 1 (inclusive)')
        if self.apx(sum([p*o*r + pi_old*p*o*(1-p)*(1-r)/(1-pi_old*p) for p, o, r in zip(pi_split, omega_split, rho_split)]))!=self.apx(rho_old*omega_old): raise ValueError('given pi_split, omega_split and rho_split do not satisfy inverse-merging constraint; retry without rho_split')
        for i in range(num_splits):
            dims[key][1].insert(block+i, names[i])
            pi[key].insert(block+i, pi_old*pi_split[i])
            rho[key].insert(block+i, rho_split[i])
            omega[key].insert(block+i, omega_split[i])
        if inplace: self.set_params({'shape':shape, 'dims':dims, 'pi':pi, 'rho':rho, 'omega':omega})
        else:
            x = self.copy()
            x.set_params({'shape':shape, 'dims':dims, 'pi':pi, 'rho':rho, 'omega':omega})
            return x

    def split(self, key, block, pi_split=None, omega_split=None, rho_split=None, names=None, inplace=True):
        key = self.get_key(key)
        block = self.get_block(key, block)
        if self.iterable(block): raise TypeError('expected exactly 1 block to be split at a time')
        if pi_split is None: pi_split = 1.0
        if not self.iterable(pi_split):
            num_splits = round(1/pi_split)
            pi_split = [1/num_splits]*num_splits
        else: num_splits = len(pi_split)
        if not all([0<i<=1 for i in pi_split]): raise ValueError('pi_split (splitting proportions) must be between 0 (exclusive) and 1 (inclusive)')
        if self.sum_pi(pi=[pi_split], approx=True)[0]!=1: raise ValueError('pi_split (splitting proportions) must sum up to 1')
        if num_splits==1: return
        shape = list(self.shape)
        shape[key] += num_splits-1
        dims = list(self.dims)
        dims[key] = list(dims[key])
        dims[key][1] = list(dims[key][1])
        dims_old = str(dims[key][1].pop(block))
        if names is None: names = dims_old
        if not self.iterable(names): names = [names+' # '+str(i+1) for i in range(num_splits)]
        if len(names)!=num_splits: raise TypeError('expected names for exactly %d new blocks'%num_splits)
        pi = list(self.pi)
        pi[key] = list(pi[key])
        pi_old = pi[key].pop(block)
        rho = list(self.rho)
        rho[key] = list(rho[key])
        rho_old = rho[key].pop(block)
        omega = list(self.omega)
        omega[key] = list(omega[key])
        omega_old = omega[key].pop(block)
        if omega_split is None: omega_split = omega_old
        if not self.iterable(omega_split): omega_split = [omega_split]*num_splits
        if len(omega_split)!=num_splits: raise ValueError('expected omega_split for exactly %d blocks'%num_splits)
        if not all([0<=i for i in omega_split]): raise ValueError('omega_split must be >=0')
        if self.mean_omega(omega=[omega_split], pi=[pi_split], approx=True, collapse=False)[0]!=self.apx(omega_old): raise ValueError('given omega_split does not satisfy mean omega constraint; retry without omega_split or with an apt one')
        coefficient = omega_old*rho_old
        if rho_split is None: rho_split = [coefficient*p/o for p, o in zip(pi_split, omega_split)]
        if not self.iterable(rho_split): rho_split = [rho_split]*num_splits
        if len(rho_split)!=num_splits: raise ValueError('expected rho_split for exactly %d blocks'%num_splits)
        if not all([0<=i<=1 for i in rho_split]): raise ValueError('rho_split must be between 0 and 1 (inclusive)')
        if self.apx(sum([o*r for o, r in zip(omega_split, rho_split)]))!=self.apx(rho_old*omega_old): raise ValueError('given pi_split, omega_split and rho_split do not satisfy splitting constraint; retry without rho_split')
        for i in range(num_splits):
            dims[key][1].insert(block+i, names[i])
            pi[key].insert(block+i, pi_old*pi_split[i])
            rho[key].insert(block+i, rho_split[i])
            omega[key].insert(block+i, omega_split[i])
        if inplace: self.set_params({'shape':shape, 'dims':dims, 'pi':pi, 'rho':rho, 'omega':omega})
        else:
            x = self.copy()
            x.set_params({'shape':shape, 'dims':dims, 'pi':pi, 'rho':rho, 'omega':omega})
            return x

    def split_inverse(self, key, blocks, name=None, inplace=True):
        key = self.get_key(key)
        blocks = self.get_block(key, blocks)
        if not blocks or not self.iterable(blocks):
            warn('expected at least 2 blocks to be merged together, no split-inverse performed for dim %s'%str(key), RuntimeWarning)
            return
        shape = list(self.shape)
        shape[key] += 1-len(blocks) 
        dims = list(self.dims)
        dims[key] = list(dims[key])
        dims[key][1] = list(dims[key][1])
        dims_new = list()
        pi = list(self.pi)
        pi[key] = list(pi[key])
        pi_new = list()
        rho = list(self.rho)
        rho[key] = list(rho[key])
        rho_new = list()
        omega = list(self.omega)
        omega[key] = list(omega[key])
        omega_new = list()
        new_idx = blocks[0]
        blocks = sorted(blocks)
        for i in range(len(blocks)):
            dims_new.append(dims[key][1].pop(blocks[i]-i))
            pi_new.append(pi[key].pop(blocks[i]-i))
            rho_new.append(rho[key].pop(blocks[i]-i))
            omega_new.append(omega[key].pop(blocks[i]-i))
        sum_pi = sum(pi_new)
        sum_om = sum([p*o for p, o in zip(pi_new, omega_new)])
        if name is None: name = ' !# '.join(['('+str(b)+')' for b in dims_new])
        dims[key][1].insert(new_idx, name)
        pi[key].insert(new_idx, sum_pi)
        omega[key].insert(new_idx, sum_om/sum_pi)
        if sum_om == 0: rho[key].insert(new_idx, 0.)
        else: rho[key].insert(new_idx, sum([o*r for o, r in zip(omega_new, rho_new)])*sum_pi/sum_om)
        if inplace: self.set_params({'shape':shape, 'dims':dims, 'pi':pi, 'rho':rho, 'omega':omega})
        else:
            x = self.copy()
            x.set_params({'shape':shape, 'dims':dims, 'pi':pi, 'rho':rho, 'omega':omega})
            return x

    def compress(self, keys=None, thresh=0.1, inplace=True):
        def foo(ego, key):
            f = []
            for i in range(ego.shape[key]):
                for j in range(i):
                    p_i = ego.pi[key][i]
                    p_j = ego.pi[key][j]                    
                    r_i = ego.rho[key][i]
                    r_j = ego.rho[key][j]
                    o_i = ego.omega[key][i]
                    o_j = ego.omega[key][j]
                    p = p_i+p_j
                    rp = p_i*o_i*(r_i+(p-p_i)*(1-r_i)/(1-p_i))
                    rp += p_j*o_j*(r_j+(p-p_j)*(1-r_j)/(1-p_j))
                    rp /= (p_i*o_i+p_j*o_j)
                    #try: rp /= (p_i*o_i+p_j*o_j)
                    #except ZeroDivisionError as err: rp = 0
                    f.append((abs(rp-(r_i*p_i+r_j*p_j))*p, i, j))
            return f
        keys = self.get_key(keys)
        if not self.iterable(keys): keys = [keys]
        tmp = self.copy()
        for key in keys:
            aff = tmp.homophily()[key]
            aff_inf = [i for i in range(len(aff)) if aff[i]==-float('inf')] #merge all perfectly heterophilous communities together
            if aff_inf: tmp.merge(key, aff_inf)
            while True:
                obj = foo(tmp, key)
                if obj: min_v, min_i, min_j = min(obj, key=lambda x: x[0])
                else: break
                if min_v<=thresh: tmp.merge(key, [min_i, min_j])
                else: break
        tmp.name = '>'+self.name+'<'
        if inplace:
            tmp_ho = tmp.mean_homophily()
            ori_ho = self.mean_homophily()
            names = self.keys()
            shape = self.shape
            self.set_params(tmp.get_params())
            return tuple([(names[key], {'loss':round(ori_ho[key]-tmp_ho[key],3), 'rate':round(1-tmp.shape[key]/shape[key],3)}) for key in keys])
        else: return tmp
    
    def get_model(self, mode='full', directed=True, name=None): return self.StochasticBlockModel(self, mode=mode, directed=directed, name=name)
    
    class StochasticBlockModel():
        
        def __init__(self, ego=None, pi=None, mode='full', directed=True, counts=False, name=None, filepath=None):
            
            if isinstance(ego, str): ego = EgocentricSBM(filepath=ego)
            elif isinstance(ego, np.ndarray):
                shape = ego.shape
                if len(shape)!=2 or shape[0]!=shape[1]: raise ValueError('expected a square block/count matrix')
                self.mode = 'global'
                self.ndim = 1
                self.shape = (shape[0],)
                if name is None or isinstance(name, str): name = (name, [str(i) for i in range(shape[0])])
                elif len(name)==shape[0] and all([isinstance(n, str) for n in name]): name = (None, name)
                elif len(name)==2 and (name[0] is None or isinstance(name[0], str)) and len(name[1])==shape[0] and all([isinstance(n, str) for n in name[1]]): pass
                else: raise ValueError('unable to properly parse dimension names')
                self.name = str(name[0])
                self.dims = tuple(name[1])
                self._precision = 8
                if pi is None: pi = 'uni'
                pi = self.get_pi(pi).flatten()
                self.pi = (tuple(pi),)
                self.params = tuple()
                if not (ego>=0).all(): raise ValueError('entries of count/block matrix must be non-negative')
                if counts:
                    warn('assuming given matrix is a count matrix to be normalised w.r.t. pi', RuntimeWarning)
                    self.psi = (ego*np.dot((1/pi)[:,np.newaxis], (1/pi)[np.newaxis,:]),)
                else:
                    warn('assuming given matrix is a block matrix; if otherwise use counts=True', RuntimeWarning)
                    self.psi = (ego.copy(),)
                self.directed = directed and (ego!=ego.transpose()).any()
                self.meanomega = 1.
                self.meanomega = self.mean_affinity()
                return
            elif ego is None:
                if filepath is None: raise ValueError('either provide "ego" as a valid EgocentricSBM, or as the path to an .ego file containing a valid EgocentricSBM, or as a square np.ndarray representing the block matrix, or provide "filepath" to an .npz file containing a valid StochasticBlockModel')
                self.load(filepath)
                if name is not None: self.name = str(name)
                return
            if mode not in ['full', 'pp', 'ppcollapsed']: raise ValueError('invalid model mode "%s"'%mode)
            if name is None: name = ego.name + '.sbm'
            self.mode = mode
            self.name = str(name)
            self.ndim = ego.ndim
            self.shape = ego.shape
            self.dims = ego.get_dims()
            self._precision = max(8, ego.precision)
            if pi is None: self.pi = ego.pi
            else:
                self.check_pi(pi)
                self.pi = tuple(pi)
            self.meanomega = ego.mean_omega()
            
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

        def get_params(self): return {'mode':self.mode, 'name':self.name, 'ndim':self.ndim, 'shape':self.shape, 'dims':self.dims, 'pi':self.pi, 'meanomega':self.meanomega, 'precision':self.precision, 'params':self.params, 'psi':self.psi, 'directed':self.directed}
            
        def __str__(self):
            params = self.get_params()
            out = [['name:', params['name']]]
            out.append(['mode:', params['mode']])
            out.append(['ndim:', str(params['ndim'])])
            out.append(['shape:', str(params['shape'])])
            out.append(['directed:', str(params['directed'])])
            out.append(['affinity:', str(params['meanomega'])])
            out.append(['pi:', str(params['pi'])])
            out.append(['params:', str(params['params'])])
            width = max([len(i[0])+4 for i in out])
            for i in range(len(out)): out[i] = out[i][0].ljust(width) + out[i][1]
            return '\n'.join(out)
    
        def __len__(self): return self.get_shape()

        def apx(self, param): return np.around(param, self.precision)

        def __eq__(self, other):
            if isinstance(other, type(self)):
                if len(self)==len(other): return (self.get_psi(approx=True, sort=True)==other.get_psi(approx=True, sort=True)).all()
                else: return False
            else: return False

        def save(self, filepath=None):
            if filepath is None: filepath = self.name
            np.savez(filepath, **self.get_params())

        def load(self, filepath):
            file = np.load(filepath)
            self.name = str(file['name'])
            self.mode = str(file['mode'])
            self.ndim = int(file['ndim'])
            self.shape = tuple(file['shape'])
            self.dims = tuple(file['dims'])
            self.params = tuple(file['params'])
            self.pi = tuple([tuple(p) for p in file['pi']])
            self.psi = tuple(file['psi'])
            self.meanomega = float(file['meanomega'])
            self.precision = int(file['precision'])
            self.directed = bool(file['directed'])

        def __copy__(self):
            from tempfile import TemporaryFile
            fd = TemporaryFile(suffix='.npz')
            self.save(fd)
            fd.seek(0)
            return type(self)(filepath=fd)

        def copy(self): return self.__copy__()
            
        def kron(self, list_of_mats, divisor=1.):
                a = np.array(list_of_mats[0])
                for i in range(1, len(list_of_mats)):
                    a = np.kron(a, np.array(list_of_mats[i])/divisor)
                return a
            
        def get_shape(self): return int(self.kron(self.shape))

        def check_pi(self, pi):
            if len(pi)==self.ndim:
                for i in self.ndim:
                    if len(pi[i])!=self.shape[i]: raise ValueError('provide pi for all blocks and in proper dimension order')
                pi = self.kron(pi)
            if len(pi)!=self.get_shape(): raise ValueError('provide full or factorised pi')
            if not all([0<=p<=1 for p in pi]): raise ValueError('pi must be between 0 and 1 (inclusive)')
            if self.apx(sum(pi))!=1: raise ValueError('pi must sum up to 1')
            return pi
        
        def get_pi(self, pi=None, approx=False):
            if pi is None: return self.kron(self.pi)
            if isinstance(pi, str):
                if pi=='uni': return (1/self.get_shape())*np.ones([self.get_shape()])
                else: raise ValueError('unknown value of pi "%s"; did you mean "uni" for uniform distribution?'%pi)
            try: #pi as matrix
                pi_shape = pi.shape
                if (len(pi_shape)==1 or len(pi_shape)==2) and pi_shape[0]==self.get_shape():
                    if (pi>=0).all() and (pi<=1).all():
                        if not (self.apx(pi.sum(0))==1).all(): raise ValueError('pi must sum up to 1')
                    else: raise ValueError('pi must be between 0 and 1 (inclusive)')
                else: raise ValueError('expected pi for exactly %d blocks'%self.get_shape())
            except AttributeError as err: pi = self.check_pi(pi)                
            except Exception as err: raise err
            if approx: return self.apx(pi)
            else: return np.array(pi)
            
        def get_psi(self, directed=None, log=False, ratio=False, approx=False, sort=False):
            if self.mode=='ppcollapsed': out = self.kron(self.psi, self.meanomega)
            elif self.mode=='pp': out = np.matmul(self.kron([x[0] for x in self.psi], self.meanomega), self.kron([x[1] for x in self.psi]))
            elif self.mode=='full': out =  np.matmul(self.kron([x[0] for x in self.psi], self.meanomega), self.kron([x[1] for x in self.psi]))
            elif self.mode=='global': out = self.psi[0]
            if directed is None: directed = self.directed
            if not directed:
                if self.directed: warn('symmetrising the stochastic block matrix', RuntimeWarning)
                out = (out + out.transpose())/2
            if sort:
                idx = np.argsort(np.diag(out), kind='mergesort')[::-1] #sort by decreasing homophily
                out = out[idx,:][:,idx]
            if ratio: out /= np.diag(out)[:,np.newaxis]
            if log: out = np.log2(out)
            if approx: out = self.apx(out)
            return out

        def get_omega(self, directed=None, approx=False): return self.find_mean(self.get_psi(directed, approx=approx), pi=None, approx=approx)

        def get_rho(self, directed=None, approx=False):
            out = np.diag(self.get_psi(directed))*self.get_pi()/self.get_omega(directed)
            if approx: out = self.apx(out)
            return out

        def get_laplacian(self, mode='out', pi=None, norm=False, sym=False):
            if mode=='out': psi = self.get_psi()
            elif mode=='in': psi = self.get_psi().transpose()
            elif mode=='sym': psi = self.get_psi(directed=False)
            else: raise ValueError('mode "%s" not recognized; use "out"/"in"/"sym"'%mode)
            if pi!=False:
                pi = self.get_pi(pi)
                psi *= np.dot(pi[:,np.newaxis], pi[np.newaxis,:])
            sum_psi = psi.sum(1)
            lap = np.diag(sum_psi) - psi #ordinary laplacian
            if sym and norm: lap = np.matmul(np.matmul(np.diag(np.sqrt(1/sum_psi)), lap), np.diag(np.sqrt(1/sum_psi))) #symmetric normalized
            elif not sym and norm: lap = np.matmul(np.diag(1/sum_psi), lap) #random walk normalized
            elif sym and not norm: raise ValueError('invalid combination of sym (%s) and norm (%s); did you mean directed=False?'%(sym, norm))
            return lap

        def get_key(self, key=None):
            if isinstance(key, int):
                shape = self.get_shape()
                if key<0 and key>=-shape: key += shape
                if key>=0 and key<shape: pass
                else: raise IndexError('sbm index out of range; must be between 0, %d (inclusive)'%(shape-1))
            elif isinstance(key, str):
                found = False
                for i in range(len(self.dims)):
                    if self.dims[i]==key:
                        key = i
                        found = True
                        break
                if not found: raise KeyError('sbm key "%s" not found; must be dimension name'%key)
            else: raise KeyError('sbm key "%s" not found; must be dimension name'%key)
            return key

        def cluster(self, mode='sym', pi=False, norm=True, k=None, seed=0, plot=True):
            #see https://www.cs.cmu.edu/~aarti/Class/10701/readings/Luxburg06_TR.pdf
            if k is None:
                from math import log2, ceil
                k = ceil(log2(len(self)))
            lap = self.get_laplacian(mode, pi, norm)
            eigval, eigvec = np.linalg.eigh(lap)
            k = max(min(len(eigvec), k), 1)
            if plot:
                try:
                    import matplotlib.pyplot as plt
                    plt.subplot(1, 2, 1)
                    plt.scatter(range(1, k+1), eigval[:k], label='eig')
                    plt.scatter(range(1, k), np.diff(eigval[:k]), label='del')
                    plt.xlabel('k')
                    plt.ylabel('eigval')
                    plt.subplot(1, 2, 2)
                    plt.scatter(range(1, len(eigval)+1), eigval, label='eig')
                    plt.scatter(range(1, len(eigval)), np.diff(eigval), label='del')
                    plt.xlabel('k')
                    plt.ylabel('eigval')
                    plt.legend()
                    plt.show()
                except: pass
            eigvec = eigvec[:,:k]
            proj = np.matmul(lap, eigvec)
            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=k, random_state=seed).fit(proj)
            return tuple(model.labels_)

        def merge(self, keys=None):
            if self.mode!='global': raise RuntimeError('cannot merge blocks for a "%s" model'%self.mode)
            if keys is None: idx = list(np.where(np.diag(self.get_psi(approx=True))==0)[0]) #merge only perfectly heterophilous communities
            else: idx = sorted([self.get_key(key) for key in set(keys)])
            if len(idx)>1:
                idx_new = idx[0]
                pi = self.get_pi()
                psi = self.get_psi()
                psi *= np.dot(pi[:,np.newaxis], pi[np.newaxis,:])
                counts_out = psi[:,idx].sum(1)
                counts_out = np.insert(np.delete(counts_out, idx), idx_new, counts_out[idx].sum())
                counts_in = np.delete(psi[idx,:].sum(0), idx)
                psi = np.insert(np.insert(np.delete(np.delete(psi, idx, 1), idx, 0), idx_new, counts_in, 0), idx_new, counts_out, 1)
                pi = np.insert(np.delete(pi, idx), idx_new, pi[idx].sum())
                psi *= np.dot((1/pi)[:,np.newaxis], (1/pi)[np.newaxis,:])
                self.pi = (tuple(pi),)
                self.psi = (psi,)
                dims = list(self.dims)
                dims_new = []
                for i in range(len(idx)): dims_new.append(dims.pop(idx[i]-i))
                dims.insert(idx_new, ' | '.join(dims_new))
                self.dims = tuple(dims)
                self.shape = (self.shape[0]-len(idx)+1,)
                self.directed = self.directed and (psi!=psi.transpose()).any()
                #self.meanomega = self.mean_affinity() #ideally this should not change!

        def compress(self, mode='sym', pi=False, norm=True, m=None, inplace=True):
            clusters = self.cluster(mode, pi, norm, m, plot=False)
            to_merge = [[] for i in range(max(clusters)+1)]
            for i in range(len(clusters)): to_merge[clusters[i]].append(self.dims[i])
            if inplace:
                for cluster in to_merge: self.merge(cluster)
            else:
                out = self.copy()
                for cluster in to_merge: out.merge(cluster)
                return out

        def project(self, directed=False, m=None, scale=False, plot=False):
            if m is None:
                from math import log2, ceil
                m = ceil(log2(len(self)))
            a = self.get_psi(directed=directed)
            eigval, eigvec = np.linalg.eig(a)
            idx = np.argsort(np.abs(eigval), kind='mergesort')
            eigval = eigval[idx][::-1]
            eigvec = eigvec[:, idx][:, ::-1]
            proj = np.dot(a, eigvec)
            proj /= np.linalg.norm(a, 2, 1)[:,np.newaxis]
            com_pos = [[self.dims[j] for j in range(proj.shape[0]) if proj[j,i]>=0] for i in range(1, m+1)]
            com_neg = [[self.dims[j] for j in range(proj.shape[0]) if proj[j,i]<0] for i in range(1, m+1)]
            psi = []
            pi = []
            om = []
            for i in range(m):
                tmp = self.copy()
                tmp.merge(com_pos[i])
                tmp.merge(com_neg[i])
                psi.append(tmp.get_psi(directed=directed))
                pi.append(tmp.get_pi())
                om.append(tmp.get_omega())
            omega_full = self.kron(om, self.meanomega)
            pi_full = self.kron(pi)
            psi_full = self.kron(psi, self.meanomega)
            if scale:
                factor = max(np.abs(eigval))/max(np.abs(np.linalg.eigvals(psi_full)))
                omega_full *= factor
                psi_full *= factor
            if plot:
                import matplotlib.pyplot as plt
                eigval_full = np.linalg.eigvals(psi_full)
                eigval_logabs = np.sort(np.log2(np.abs(eigval)), kind='mergesort')
                eigval_full_logabs = np.sort(np.log2(np.abs(eigval_full)), kind='mergesort')
                eigval_del = np.diff(eigval_logabs)
                h1 = plt.scatter(eigval_logabs[1:], eigval_del, color='black', label='original')
                for i in eigval_full_logabs[-(m+1):]: h2 = plt.gca().axvline(i, ls='--', color='red', label='inferred (top m+1)')
                plt.xlabel('log(abs(eig))')
                plt.ylabel('del log(abs(eig))')
                plt.legend(handles=[h1, h2])
                plt.title('inferring latent sbm with %d dims'%m)
                plt.show()
                try:                    
                    import seaborn as sns
                    omo = np.log2(self.get_omega())
                    omo = omo[~np.isinf(omo)]
                    omi = np.log2(omega_full)
                    omi = omi[~np.isinf(omi)]
                    sns.distplot(omo, label='original')
                    sns.distplot(omi, label='inferred')
                    plt.gca().axvline(np.log2(self.meanomega), ls='--', color='black', label='original mean omega')
                    plt.title('distribution of log(omega)')
                    plt.legend()
                    plt.show()
                    sns.distplot(eigval_logabs, label='original')
                    sns.distplot(eigval_full_logabs, label='inferred')
                    plt.legend()
                    plt.title('distribution of log(abs(eig))')
                    plt.show()
                except ImportError as err: pass
                except Exception as err: raise err
            return type(self)(psi_full, pi_full, name=self.name+'.proj')

        def eigproj(self, directed=False, m=None, plot=False):
            if m is None:
                from math import log2, ceil
                m = ceil(log2(len(self)))
            a = self.get_psi(directed=directed)
            eigval, eigvec = np.linalg.eig(a)
            eigval_logabs = np.log2(np.abs(eigval))
            idx = np.argsort(eigval_logabs, kind='mergesort')
            eigval_sorted = eigval[idx]
            eigval_logabs = eigval_logabs[idx]
            eigvec = eigvec[:, idx][:, ::-1]
            proj = np.dot(a, eigvec)
            proj /= np.linalg.norm(a, 2, 1)[:,np.newaxis]
            com_pos = [[self.dims[j] for j in range(proj.shape[0]) if proj[j,i]>=0] for i in range(1, m+1)]
            com_neg = [[self.dims[j] for j in range(proj.shape[0]) if proj[j,i]<0] for i in range(1, m+1)]
            pi = []
            for i in range(m):
                tmp = self.copy()
                tmp.merge(com_pos[i])
                tmp.merge(com_neg[i])
                pi.append(tmp.get_pi())
            pi_full = self.kron(pi)
            a = np.ones((m, m)) - np.eye(m)
            b = eigval_logabs[-(m+1):-1]
            a_inv = np.linalg.inv(a)
            a_inv_b = np.dot(a_inv, b)
            k = (np.dot(np.ones(m), a_inv_b) - eigval_logabs[-1])/np.dot(np.ones(m), np.dot(a_inv, np.ones(m)))
            lambda_pos = np.dot(a_inv, b-k)
            lambda_pos = 2**lambda_pos
            lambda_neg = 2**k*np.ones(m)
            if eigval_sorted[-1]<0: raise RuntimeError('largest eigenvalue must not be negative')
            for i in range(m):
                if eigval_sorted[-(i+2)]<0: lambda_neg[-(i+1)] = -lambda_neg[-(i+1)]
            p = (lambda_pos+lambda_neg)/2
            q = (lambda_pos-lambda_neg)/2
            psi_full = self.kron([np.array([[i,j],[j,i]]) for (i, j) in zip(p, q)])
            if plot:
                import matplotlib.pyplot as plt
                eigs_psi = np.linalg.eigvalsh(psi_full)
                eigs_psi_logabs = np.log2(np.abs(eigs_psi))
                idx = np.argsort(eigs_psi_logabs, kind='mergesort')
                eigs_psi_sorted = eigs_psi[idx]
                eigs_psi_logabs = eigs_psi_logabs[idx]
                eigs_del = np.diff(eigval_sorted)
                h1 = plt.scatter(eigval_sorted[1:], eigs_del, color='black', label='original')
                for i in eigs_psi_sorted[-(m+1):]: h2 = plt.gca().axvline(i, ls='--', color='red', label='inferred (top m+1)')
                plt.xlabel('eig')
                plt.ylabel('del eig')
                plt.legend(handles=[h1, h2])
                plt.title('inferring latent sbm with %d dims'%m)
                plt.show()
                eigs_del = np.diff(eigval_logabs)
                h1 = plt.scatter(eigval_logabs[1:], eigs_del, color='black', label='original')
                for i in eigs_psi_logabs[-(m+1):]: h2 = plt.gca().axvline(i, ls='--', color='red', label='inferred (top m+1)')
                plt.xlabel('log(abs(eig))')
                plt.ylabel('del log(abs(eig))')
                plt.legend(handles=[h1, h2])
                plt.title('inferring latent sbm with %d dims'%m)
                plt.show()
                try:
                    import seaborn as sns
                    plt.figure()
                    sns.distplot(eigs_logabs, label='original')
                    sns.distplot(eigs_psi_logabs, label='inferred')
                    plt.legend()
                    plt.title('distribution of log(abs(eig))')
                    plt.show()
                except: pass
            return type(self)(psi_full, pi_full, name=self.name+'.proj')

        def find_mean(self, value, pi=None, approx=False): return np.dot(value, self.get_pi(pi, approx))

        def find_var(self, value, pi=None, mean=None, approx=False):
            if mean is None: mean = self.find_mean(value, pi, approx)
            try:
                if len(mean.shape)==2: mean = np.diag(mean) #for multiple pi
            except: pass
            try:
                if len(value.shape)==1: value = np.vstack([value]*mean.shape[0]) #for static SAS
                if len(mean.shape)==1: mean = mean[:,np.newaxis] #for apt broadcasting
            except: pass
            var = self.find_mean((value - mean)**2, pi, approx)
            try:
                if len(var.shape)==2: var = np.diag(var)
            except: pass
            return var

        def find_covar(self, value=1, pi=None, approx=False):
            pi = self.get_pi(pi, approx)
            if len(pi.shape)==1: return np.dot(value, np.diag(pi)-np.dot(pi[:,np.newaxis], pi[np.newaxis,:])) #a covariance matrix
            else:
                cov = []
                for i in range(pi.shape[1]):
                    cov.append(np.dot(value, np.diag(pi[:,i])-np.dot(pi[:,i][:,np.newaxis], pi[:,i][np.newaxis,:])))
                return np.dstack(cov)

        def generate_people(self, n, pi=None): return np.random.multinomial(1, self.get_pi(pi), n)
        
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
            return a, p, keep
        
        def generate_networkdata(self, n, pi=None, directed=False, name=None):
            z = self.generate_people(n, pi)
            a, p, keep = self.generate_network(z, directed)
            d = np.array(self.dims)
            d = tuple(d[keep])
            if name is None: name = self.name + '.net'
            return NetworkData(a, z[:,keep], d, p, name)

        def generate_networkx(self, n, pi=None, directed=False, selfloops=False):
            if not isinstance(n, int): raise TypeError('expected n to be int; got %s'%type(n))
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
            except Exception as err:
                warn('unable to import stochastic_block_model from networkx; returning dict of stochastic_block_model params', ImportWarning)
                return params
        
        def eigvals_pipsi(self, pi=None, directed=None, real=False):
            pi = self.get_pi(pi)
            if real: return sorted(np.real(np.linalg.eigvals(np.matmul(np.diag(pi), self.get_psi(directed)))), reverse=True)
            else: return sorted(np.linalg.eigvals(np.matmul(np.diag(pi), self.get_psi(directed))), reverse=True)
        
        def eigvals_pipsi_theoretical(self):
            pi = 1/self.get_shape()
            if self.mode=='full' or self.mode=='global': raise RuntimeError('cannot compute theoretical eigenvalues for a "%s" model'%self.mode)
            elif self.mode=='pp': eigs = sorted(tuple(pi*self.meanomega*self.kron([(h[0]+h[1]*(s-1),)+(h[0]-h[1],)*(s-1) for (h, s) in zip(self.params[1], self.shape)])), reverse=True)
            elif self.mode=='ppcollapsed': eigs = sorted(tuple(pi*self.meanomega*self.kron([(h[0]+h[1]*(s-1),)+(h[0]-h[1],)*(s-1) for (h, s) in zip(self.params, self.shape)], self.meanomega)), reverse=True)
            return eigs
        
        def homoffinity_out(self, pi=None, directed=None, log=False, ratio=False, approx=False): return np.diag(self.get_psi(directed, log, ratio, approx))*self.get_pi(pi, approx)
        def homoffinity_in(self, pi=None, directed=None, log=False, ratio=False, approx=False): return self.homoffinity_out(pi, directed, log, ratio, approx)
        def homoffinity(self, pi=None, directed=None, log=False, ratio=False, approx=False): return self.homoffinity_out(pi, directed, log, ratio, approx)
            
        def heteroffinity_out(self, pi=None, directed=None, log=False, ratio=False, approx=False): return self.affinity_out(pi, directed, log, ratio, approx)-self.homoffinity_out(pi, directed, log, ratio, approx)
        def heteroffinity_in(self, pi=None, directed=None, log=False, ratio=False, approx=False): return self.affinity_in(pi, directed, log, ratio, approx)-self.homoffinity_in(pi, directed, log, ratio, approx)
        def heteroffinity(self, pi=None, directed=None, log=False, ratio=False, approx=False): return self.affinity(pi, directed, log, ratio, approx)-self.homoffinity(pi, directed, log, ratio, approx)
        
        def affinity_out(self, pi=None, directed=None, log=False, ratio=False, approx=False): return self.find_mean(self.get_psi(directed, log, ratio, approx), pi, approx)
        def affinity_in(self, pi=None, directed=None, log=False, ratio=False, approx=False): return self.find_mean(self.get_psi(directed, log, ratio, approx).transpose(), pi, approx)
        def affinity(self, pi=None, directed=None, log=False, ratio=False, approx=False):
            if self.directed and (directed or directed is None): raise RuntimeError('affinity is ambiguous for directed SBMs; use affinity_in() or affinity_out()')
            else: return self.affinity_out(pi, directed, log, ratio, approx)
        
        def mean_homoffinity_out(self, pi=None, directed=None, log=False, ratio=False, approx=False): return self.find_mean(self.homoffinity_out(pi, directed, log, ratio, approx), pi, approx)
        def mean_homoffinity_in(self, pi=None, directed=None, log=False, ratio=False, approx=False): return self.find_mean(self.homoffinity_in(pi, directed, log, ratio, approx), pi, approx)
        def mean_homoffinity(self, pi=None, directed=None, log=False, ratio=False, approx=False): return self.find_mean(self.homoffinity(pi, directed, log, ratio, approx), pi, approx)
        
        def mean_heteroffinity_out(self, pi=None, directed=None, log=False, ratio=False, approx=False): return self.find_mean(self.heteroffinity_out(pi, directed, log, ratio, approx), pi, approx)
        def mean_heteroffinity_in(self, pi=None, directed=None, log=False, ratio=False, approx=False): return self.find_mean(self.heteroffinity_in(pi, directed, log, ratio, approx), pi, approx)
        def mean_heteroffinity(self, pi=None, directed=None, log=False, ratio=False, approx=False): return self.find_mean(self.heteroffinity_out(pi, directed, log, ratio, approx), pi, approx)
        
        def mean_affinity_out(self, pi=None, directed=None, log=False, ratio=False, approx=False): return self.find_mean(self.affinity_out(pi, directed, log, ratio, approx), pi, approx)
        def mean_affinity_in(self, pi=None, directed=None, log=False, ratio=False, approx=False): return self.find_mean(self.affinity_in(pi, directed, log, ratio, approx), pi, approx)
        def mean_affinity(self, pi=None, directed=None, log=False, ratio=False, approx=False): return self.find_mean(self.affinity_out(pi, directed, log, ratio, approx), pi, approx)
        
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
        
        def sas_individual_pw(self, log=True, ratio=True, metric_type=None, approx=False): return self.dis2met(-self.get_psi(log=log, ratio=ratio, approx=approx), metric_type)
        def sas_individual_out(self, log=True, ratio=True, metric_type=None, pi=None, approx=False): return self.find_mean(self.sas_individual_pw(log, ratio, metric_type, approx), pi, approx)
        def sas_individual_in(self, log=True, ratio=True, metric_type=None, pi=None, approx=False): return self.find_mean(self.sas_individual_pw(log, ratio, metric_type, approx).transpose(), pi, approx)
        def sas_individual(self, log=True, ratio=True, metric_type=None, pi=None, approx=False):
            if self.directed: raise RuntimeError('individual SAS is ambiguous for directed SBMs; use sas_individual_out() or sas_individual_in()')
            else: return self.sas_individual_out(log, ratio, metric_type, pi, approx)
        def sas_global(self, log=True, ratio=True, metric_type=None, pi=None, static=False, approx=False):
            if static: sas = self.sas_individual_out(log, ratio, metric_type, None, approx)
            else: sas = self.sas_individual_out(log, ratio, metric_type, pi, approx)
            #if (not log) and ratio: sas = -np.log2(-sas)
            sas_mean = self.find_mean(sas.transpose(), pi, approx)
            try:
                if len(sas_mean.shape)==2: sas_mean = np.diag(sas_mean)
            except: pass
            sas_var = self.find_var(sas.transpose(), pi, sas_mean, approx)
            return sas_mean, sas_var

        def generate_barcode(self, pi=None, n=float('inf'), name=None): return self.Barcode(self, pi, n, name)

        class Barcode():

            def __init__(self, sbm=None, pi=None, n=float('inf'), name=None, filepath=None):
                if isinstance(sbm, str): sbm = EgocentricSBM.StochasticBlockModel(filepath=sbm)
                elif sbm is None:
                    if filepath is None: raise ValueError('either provide "sbm" as a valid StochasticBlockModel, or as the path to an .npz file containing a valid StochasticBlockModel, or provide "filepath" to an .npz file containing a valid Barcode')
                    self.load(filepath)
                    if name is not None: self.name = str(name)
                    return
                k = len(sbm)
                idx = np.triu_indices(k)
                psi = -sbm.get_psi(directed=False, approx=True) #distances for Veitoris-Rips complex
                psi = psi[idx]
                num = len(psi)
                idx_sort = np.argsort(psi, kind='mergesort')
                psi = psi[idx_sort]
                idx_row = idx[0][idx_sort]
                idx_col = idx[1][idx_sort]
                path_matrix = np.zeros((k, k), dtype=bool)
                edge_list = list()
                conn_list = list()
                for i in range(num):
                    p = idx_row[i]
                    q = idx_col[i]
                    edge_list.append(((p, q),))
                    if p==q: conn_list.append((path_matrix[p,:].any(),))
                    else: conn_list.append(((path_matrix[p,:].any(), path_matrix[q,:].any(), path_matrix[p,q]), ))
                    path_matrix[p,q] = True #p can reach q
                    path_matrix[q,p] = True #q can reach p
                    path_matrix[p] |= path_matrix[q] #everything that can be reached by q can be reached by p
                    path_matrix[q] |= path_matrix[p] #everything that can be reached by p can be reached by q
                    path_matrix[path_matrix[p]] |= path_matrix[p] #everything that can reach p can reach everything reached by p
                    path_matrix[path_matrix[q]] |= path_matrix[q] #everything that can reach q can reach everything reached by q
                del_psi = np.diff(psi)
                keep = np.append(del_psi!=0, True)
                if not keep.all():
                    psi = psi[keep]
                    dummy = 0
                    for i in range(num):
                        if not keep[i]:
                            edge_list[i-dummy] += edge_list.pop(i-dummy+1)
                            conn_list[i-dummy] += conn_list.pop(i-dummy+1)
                            dummy += 1
                if name is None: name = sbm.name + '.bar'
                self.name = str(name)
                self.k = k
                self.dims = sbm.dims
                self.epsilon = psi
                self.event = tuple(edge_list)
                self.state = tuple(conn_list)
                self._n = float(n)
                self._pi = sbm.get_pi(pi)
                self.reset()

            def __len__(self): return len(self.epsilon)

            def set(self, pi=None, n=None):
                if pi is None: pi = self.pi
                if n is None: n = self.n
                pi = np.array(pi)
                if len(pi.shape)==0 or len(pi.shape)>2: raise ValueError('expected pi to be a 1d or 2d matrix')
                elif len(pi.shape)==2:
                    num_pi = pi.shape[1]
                    if num_pi==1: pi = pi.flatten()
                else: num_pi = 1
                if pi.shape[0]!=self.k: raise ValueError('expected pi to have %d entries, but it contains %d entries'%(self.k, pi.shape[0]))
                num = len(self)
                if num_pi==1:
                    complex_1 = 0 #number of 1-complexes (edges)
                    betti_0 = 1 #Betti number 0 (number of connected components)
                    betti_0_curve = np.zeros(num, dtype=np.float32)
                    betti_1_curve = np.zeros(num, dtype=np.float32)
                else:
                    complex_1 = np.zeros(num_pi, dtype=np.float32)
                    betti_0 = np.ones(num_pi, dtype=np.float32)
                    betti_0_curve = np.zeros((num, num_pi), dtype=np.float32)
                    betti_1_curve = np.zeros((num, num_pi), dtype=np.float32)
                if n==float('inf'):
                    for i in range(num):
                        for j in range(len(self.event[i])):
                            p, q = self.event[i][j]
                            if p==q: #intra-community closure
                                complex_1 += pi[p]**2/2 #increment number of edges
                                if not self.state[i][j]: betti_0 -= pi[p] #decrease number of connected components
                                else: pass #p was already in a larger component
                            else:  #inter-community closure
                                complex_1 += pi[p]*pi[q] #increment number of edges
                                if not self.state[i][j][0]: betti_0 -= pi[p]
                                if not self.state[i][j][1]: betti_0 -= pi[q]
                        betti_0_curve[i] = betti_0
                        betti_1_curve[i] = complex_1
                else:
                    n = float(n)
                    if n<=0: raise ValueError('expected n to be a positive numeric')
                    if (n*pi<10).any(): warn('unexpected output possible when n*pi_i>>1 does not hold for some community i', RuntimeWarning)
                    complex_0 = n #number of 0-complexes (nodes)
                    betti_0 *= n #Betti number 0 (number of connected components)
                    for i in range(num):
                        for j in range(len(self.event[i])):
                            p, q = self.event[i][j]
                            if p==q: #intra-community closure
                                complex_1 += n*pi[p]*(n*pi[p]-1)/2 #increment number of edges
                                if not self.state[i][j]: betti_0 -= n*pi[p]-1 #decrease number of connected components
                                else: pass #p was already in a larger component
                            else:  #inter-community closure
                                complex_1 += n**2*pi[p]*pi[q] #increment number of edges
                                if self.state[i][j][0] and self.state[i][j][1]:
                                    if not self.state[i][j][2]: betti_0 -= 1 #decrease number of connected components only if p-q were previously in different connected components
                                elif (not self.state[i][j][0]) and self.state[i][j][1]: betti_0 -= n*pi[p] #p gets absorbed in component of q
                                elif self.state[i][j][0] and (not self.state[i][j][1]): betti_0 -= n*pi[q] #q gets absorbed in component of p
                                else: betti_0 -= n*(pi[p]+pi[q])-1 #p and q form their own connected component
                        betti_0_curve[i] = betti_0
                        betti_1_curve[i] = betti_0 - complex_0 + complex_1 #from Euler's characteristic: k-n+m
                self.n = n
                self.pi = pi
                self.betti0 = betti_0_curve
                self.betti1 = betti_1_curve

            def reset(self): self.set(self._pi, self._n)

            def get_params(self): return {'name':self.name, 'k':self.k, 'dims':self.dims, 'epsilon':self.epsilon, 'event':self.event, 'state':self.state, '_n':self._n, '_pi':self._pi, 'n':self.n, 'pi':self.pi, 'betti0':self.betti0, 'betti1':self.betti1}

            def save(self, filepath=None):
                if filepath is None: filepath = self.name
                np.savez(filepath, **self.get_params())

            def load(self, filepath):
                file = np.load(filepath)
                self.name = str(file['name'])
                self.k = int(file['k'])
                self.dims = tuple(file['dims'])
                self.epsilon = file['epsilon']
                self.event = tuple(file['event'])
                self.state = tuple(file['state'])
                self._n = float(file['_n'])
                self._pi = file['_pi']
                self.n = float(file['n'])
                self.pi = file['pi']
                self.betti0 = file['betti0']
                self.betti1 = file['betti1']

            def delta(self, x): return np.diff(x, axis=0)

            def derivative(self, y, x):
                dy = self.delta(y)
                dx = self.delta(x)
                if len(dy.shape)==1 and len(dx.shape)==2: dy = dy[:, np.newaxis]
                elif len(dy.shape)==2 and len(dx.shape)==1: dx = dx[:, np.newaxis]
                return dy/dx

            def integrate(self, y, x): return np.trapz(y, x, axis=0)

            def factorize(self, delimiter_dim=',', delimiter_val=':'):
                try: dims = [dict([tuple(d.split(delimiter_val)) for d in dim.split(delimiter_dim)]) for dim in self.dims]
                except ValueError as err: raise ValueError('unable to factorize dimension names; ensure apt delimiter_dim and delimiter_val')
                dims_dict = {k:list() for k in dims[0].keys()}
                for i in range(len(dims)):
                    try:
                        for k in dims[i].keys(): dims_dict[k].append(dims[i][k])
                    except: raise ValueError('unable to factorize dimension names; ensure identical dimension names')
                return {k:tuple(dims_dict[k]) for k in dims_dict}

            def get_event(self, key='', exact=False, selfjoin=False):
                if key:
                    if exact:
                        if selfjoin: out = [tuple([(i, self.dims[x[0]]) for x in e if (key == self.dims[x[0]] and self.dims[x[0]] == self.dims[x[1]])]) for e, i in zip(self.event, range(len(self)))]
                        else: out = [tuple([(i, self.dims[x[0]], self.dims[x[1]]) for x in e if (key == self.dims[x[0]] or key == self.dims[x[1]])]) for e, i in zip(self.event, range(len(self)))]
                    else:
                        if isinstance(key, str): key = [key]
                        if selfjoin: out = [tuple([(i, self.dims[x[0]]) for x in e if (all([k in self.dims[x[0]] for k in key]) and self.dims[x[0]] == self.dims[x[1]])]) for e, i in zip(self.event, range(len(self)))]
                        else: out = [tuple([(i, self.dims[x[0]], self.dims[x[1]]) for x in e if (all([k in self.dims[x[0]] for k in key]) or all([k in self.dims[x[1]] for k in key]))]) for e, i in zip(self.event, range(len(self)))]
                    return tuple([x for x in out if x])
                else: return tuple([tuple([(i, self.dims[x[0]], self.dims[x[1]]) for x in e]) for e, i in zip(self.event, range(len(self)))])

            def get_history(self):
                events = self.get_event()
                return '\n'.join(['['+str(e[0][0])+'] '+' & '.join([x[1]+' <---> '+x[2] for x in e]) for e in events])

            def get_epoch(self, norm=False):
                l = len(self)
                if norm: return np.arange(l)/(l-1)
                else: return np.arange(l)

            def get_epsilon(self, norm=False, log=False, plot=False):
                out = self.epsilon.copy()
                if log: out = -np.log2(1+np.abs(out))
                if norm: out = (out - out.min())/(out.max() - out.min())
                if plot:
                    import seaborn as sns
                    sns.distplot(out).set_title('distribution of epsilon; norm=%s, log=%s'%(norm, log))
                return out

            def get_betti(self, n=0, norm=False):
                if n==0: out = self.betti0.copy()
                elif n==1: out = self.betti1.copy()
                else: raise ValueError('expected n to be 0 or 1 but got %s instead'%str(n))
                if norm:
                    if self.n==float('inf'):
                        if n==1: out *= 2
                    else:
                        if n==0:
                            out -= 1
                            out /= self.n-1
                        elif n==1: out /= (self.n-1)*(self.n-2)/2
                return out

            def del_betti(self, n=0, norm=False): return self.derivative(self.get_betti(n, norm), self.get_epsilon(norm))

            def clip(self, f, top_k=0, bottom_k=0):         
                if top_k:
                    max_f = np.partition(f, -(top_k+1), axis=0)[-(top_k+1)]
                    if max_f.shape==(): f[f>max_f] = max_f
                    else:
                        for i in range(len(max_f)):
                            f[f[:,i]>max_f[i], i] = max_f[i]
                if bottom_k:
                    min_f = np.partition(f, bottom_k, axis=0)[bottom_k]
                    if min_f.shape==(): f[f<min_f] = min_f
                    else:
                        for i in range(len(min_f)):
                            f[f[:,i]<min_f[i], i] = min_f[i]
                
            def plot(self, norm=False, clip_k=0, log=False):
                import matplotlib.pyplot as plt
                x = self.get_epsilon(norm, log)
                epoch = self.get_epoch(norm)
                betti0 = self.get_betti(0, norm)
                betti1 = self.get_betti(1, norm)
                del_betti0 = self.del_betti(0, norm)
                del_betti1 = self.del_betti(1, norm)
                if clip_k:
                    self.clip(del_betti0, clip_k, clip_k)
                    self.clip(del_betti1, clip_k, clip_k)
                plt.figure(dpi=90)
                plt.subplot(2, 3, 1)
                plt.plot(x, betti0)
                plt.xlabel('epsilon')
                plt.ylabel('betti0')
                plt.subplot(2, 3, 2)
                plt.plot(x[:-1], del_betti0)
                plt.xlabel('epsilon')
                plt.ylabel('del betti0 / del epsilon')
                plt.subplot(2, 3, 3)
                plt.plot(epoch, betti0)
                plt.xlabel('epoch')
                plt.ylabel('betti0')
                plt.subplot(2, 3, 4)
                plt.plot(x, betti1)
                plt.xlabel('epsilon')
                plt.ylabel('betti1')
                plt.subplot(2, 3, 5)
                plt.plot(x[:-1], del_betti1)
                plt.xlabel('epsilon')
                plt.ylabel('del betti1 / del epsilon')
                plt.subplot(2, 3, 6)
                plt.plot(epoch, betti1)
                plt.xlabel('epoch')
                plt.ylabel('betti1')
                plt.figure(dpi=90)
                plt.plot(epoch, x)
                plt.xlabel('epoch')
                plt.ylabel('epsilon')
                plt.tight_layout()
                plt.show()

            def plot_dist(self, by='', norm=True, log=False, delimiter_dim=',', delimiter_val=':'):
                if by: names = self.factorize(delimiter_dim, delimiter_val)[by]
                else: names = self.dims
                names_set = list(set(names))
                import matplotlib.pyplot as plt                
                import seaborn as sns
                epsilon = self.get_epsilon(norm, log)
                epoch = self.get_epoch(norm)
                epsilon_curve = list()
                epoch_curve = list()
                for name in names_set:
                    a, b = zip(*[(epsilon[i], epoch[i]) for i in range(len(self)) if any([(names[j[0]]==name) | (names[j[1]]==name) for j in self.event[i]])])
                    epsilon_curve.append(a)
                    epoch_curve.append(b)
                for name, epsilon in zip(names_set, epsilon_curve): sns.distplot(epsilon, hist=False, label=name)
                plt.xlabel('epsilon')
                plt.ylabel('density')
                plt.title(by)
                plt.legend()
                plt.show()
                for name, epoch in zip(names_set, epoch_curve): sns.distplot(epoch, hist=False, label=name)
                plt.xlabel('epoch')
                plt.ylabel('density')
                plt.title(by)
                plt.legend()
                plt.show()

            def area(self, epsilon=True, norm=True, log=False):
                if epsilon: x = self.get_epsilon(norm, log)
                else: x = self.get_epoch(norm)
                betti0 = self.get_betti(0, norm)
                betti1 = self.get_betti(1, norm)
                return self.integrate(betti0, x), betti1[0] + self.integrate(betti1, x)
        
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

    def get_params(self):
        return {'name':self.name, 'names':self.names, 'n':self.n, 'k':self.k, 'directed':self.directed, 'adj':self.adj, 'mem':self.mem, 'p':self.p}

    def __str__(self):
        out = [['name:', self.name]]
        out.append(['n:', str(self.n)])
        out.append(['k:', str(self.k)])
        out.append(['directed:', str(self.directed)])
        out.append(['mean degree:', str(self.mean_degree())])
        width = max([len(i[0])+4 for i in out])
        for i in range(len(out)): out[i] = out[i][0].ljust(width) + out[i][1]
        return '\n'.join(out)

    def __len__(self): return self.n

    def __eq__(self, other):
        if isinstance(other, type(self)):
            if len(self)==len(other): return (self.adj==other.adj).all()
            else: return False
        else: return False

    def save(self, filepath=None):
        if filepath is None: filepath = self.name
        np.savez(filepath, **self.get_params())

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

    def __copy__(self):
        from tempfile import TemporaryFile
        fd = TemporaryFile(suffix='.npz')
        self.save(fd)
        fd.seek(0)
        return type(self)(filepath=fd)

    def copy(self): return self.__copy__()

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