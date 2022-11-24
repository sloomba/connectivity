import networkx as nx
import numpy as np
from .utils import *
from .egosbm import EgocentricSBM as esbm
from .probs import ShortestPathDistribution as spd

def map_snap(name=None, datapath=True, features=False):
    DATA_DICT = {
            'bitcoin': ('bitcoin/soc-sign-bitcoinotc.csv', True),
            'bitcoin_alpha': ('bitcoin/soc-sign-bitcoinalpha.csv', True),
            'collab_1': ('collab/CA-GrQc.txt', False),
            'collab_2': ('collab/CA-HepTh.txt', False),
            'college': ('college/CollegeMsg.txt', True),
            'euemail': ('euemail/email-Eu-core.txt', True),
            'fb': ('fb/facebook_combined.txt', False),
            'fb_pages': ('fb_pages/musae_facebook_edges.csv', False),
            'git': ('git/musae_git_edges.csv', False),
            'gnutella_1': ('gnutella/p2p-Gnutella04.txt', True),
            'gnutella_2': ('gnutella/p2p-Gnutella05.txt', True),
            'gnutella_3': ('gnutella/p2p-Gnutella06.txt', True),
            'gnutella_4': ('gnutella/p2p-Gnutella08.txt', True),
            'gnutella_5': ('gnutella/p2p-Gnutella09.txt', True),
            'lastfm': ('lastfm/lastfm_asia_edges.csv', False),
            'twitch_de': ('twitch/DE/musae_DE_edges.csv', False),
            'twitch_en': ('twitch/ENGB/musae_ENGB_edges.csv', False),
            'twitch_es': ('twitch/ES/musae_ES_edges.csv', False),
            'twitch_fr': ('twitch/FR/musae_FR_edges.csv', False),
            'twitch_pt': ('twitch/PTBR/musae_PTBR_edges.csv', False),
            'twitch_ru': ('twitch/RU/musae_RU_edges.csv', False),
            'twitter': ('twitter/twitter_combined.txt', True),
            'wiki': ('wiki/wikispeedia_paths-and-graph/links.tsv', True),
            'wiki_vote': ('wiki/Wiki-Vote.txt', True)}

    NAME_DICT = {
        'bitcoin': 'soc-sign-bitcoin-otc',
        'bitcoin_alpha': 'soc-sign-bitcoin-alpha',
        'collab_1': 'ca-GrQc',
        'collab_2': 'ca-HepTh',
        'college': 'CollegeMsg',
        'euemail': 'email-Eu-core',
        'fb': 'ego-Facebook',
        'fb_pages': 'musae-facebook',
        'git': 'musae-github',
        'gnutella_1': 'p2p-Gnutella04',
        'gnutella_2': 'p2p-Gnutella05',
        'gnutella_3': 'p2p-Gnutella06',
        'gnutella_4': 'p2p-Gnutella08',
        'gnutella_5': 'p2p-Gnutella09',
        'lastfm': 'feather-lastfm-social',
        'twitch_de': 'musae-twitch-DE',
        'twitch_en': 'musae-twitch-EN',
        'twitch_es': 'musae-twitch-ES',
        'twitch_fr': 'musae-twitch-FR',
        'twitch_pt': 'musae-twitch-PT',
        'twitch_ru': 'musae-twitch-RU',
        'twitter': 'ego-Twitter',
        'wiki': 'wikispeedia',
        'wiki_vote': 'wiki-Vote'
    }

    FEAT_DICT = {'euemail': ('euemail/email-Eu-core-department-labels.txt', 'dep')}

    if datapath:
        if features: raise ValueError('only one of "datapath" or "features" can be true')
        if name is None: return DATA_DICT
        try: return DATA_DICT[name]
        except: raise ValueError('snap network with name "%s" not found'%str(name))
    elif features:
        if name is None: return FEAT_DICT
        try: return FEAT_DICT[name]
        except: raise ValueError('snap network with name "%s" not found'%str(name))
    else:
        if name is None: return NAME_DICT
        try: return NAME_DICT[name]
        except: raise ValueError('snap network with name "%s" not found'%str(name))

def import_network(name='all', filepath='', dirpath='./snap', directed=None, to_undirected=False, gcc=True, modmax=False, cutoff=0.01, verbose=True):
    from networkx.algorithms.community import greedy_modularity_communities as modmax
    import os
    if name=='all': return {key: import_network(key, filepath, dirpath, directed, to_undirected, gcc, modmax, cutoff, verbose) for key in map_snap()}
    if filepath:
        if directed is None: directed = False
    else: filepath, directed = map_snap(name)
    if directed: G = nx.DiGraph()
    else: G = nx.Graph()
    if filepath[-4:]=='.csv':
        with open(os.path.join(dirpath, filepath), 'r') as fd: edgelist = [tuple([int(j) if j.isdigit() else j for j in i.strip().split(',')[:2]]) for i in fd.readlines() if '#' not in i]
    else:
        with open(os.path.join(dirpath, filepath), 'r') as fd: edgelist = [tuple([int(j) if j.isdigit() else j for j in i.strip().split()[:2]]) for i in fd.readlines() if '#' not in i]
    edgelist = [i for i in edgelist if i and i[0]!=i[1]]
    G.add_edges_from(edgelist)
    node_feat = map_snap(datapath=False, features=True)
    if name in node_feat:
        filepath, label = node_feat[name]
        with open(os.path.join(dirpath, filepath), 'r') as fd: featlist = [tuple([int(j) if j.isdigit() else j for j in i.strip().split()[:2]]) for i in fd.readlines() if '#' not in i]
        for (a, b) in featlist:
            try: G.nodes[a][label] = b
            except: pass
    else: label=''
    if to_undirected: G = G.to_undirected()
    if gcc:
        gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        for i in range(1, len(gcc)): G.remove_nodes_from(gcc[i])
    def get_community(G):
        Gcoms = modmax(G)
        size = cutoff*len(G)
        Gc = []
        outliers = set()
        for i in range(len(Gcoms)):
            if len(Gcoms[i])>=size: Gc.append(Gcoms[i])
            else: outliers.update(Gcoms[i])
        if len(outliers)>size: Gc.append(outliers)
        else: G.remove_nodes_from(outliers)
        sizes = [len(i) for i in Gc]
        if verbose: print('Inferred community sizes of %i nodes: '%sum(sizes), sizes)
        for i in range(len(Gc)):
            for j in Gc[i]: G.nodes[j]['com'] = i
    if modmax: get_community(G)
    return G

def import_netsbm(name='all', dirpath='./snap/out', label=True, infer=False, cutoff=64, verbose=True):
    import pandas as pd
    import os
    if name=='all': return {key: import_netsbm(key, dirpath, label, infer, cutoff, verbose) for key in map_snap()}
    nodes = pd.read_csv(os.path.join(dirpath, name+'_nodes.csv'), index_col=0)
    edges = pd.read_csv(os.path.join(dirpath, name+'_edges.csv'))
    block = list(np.load(os.path.join(dirpath, name+'_block.npz')).values())
    block = {'blk_%i'%(i+1):block[i] for i in range(len(block)-1)}
    G = nx.Graph()
    G.add_edges_from(edges.values)
    atts = [i for i in list(nodes) if 'blk' in i]
    for v in nodes.index:
        for att in atts: G.nodes[v][att] = nodes[att][v]
    models = {}
    models_c = {}
    for att in atts:
        pi = nodes.groupby(att).size().values
        if len(pi)<=cutoff:
            if infer:
                pi = pi/pi.sum()
                counts = block[att]
                for i in range(len(counts)): counts[i,i] /= 2
                psi = counts*np.dot((1/pi)[:,np.newaxis], (1/pi)[np.newaxis,:])/len(G)
                model = esbm.StochasticBlockModel(psi, pi)
                if verbose: print('%s | sbm mean degree: %.2f'%(att, model.meanomega))
                spdist = spd(model.get_psi(), len(G), model.get_pi())
                models[att] = {'sbm': model, 'spd': spdist}
            if label: models_c[att] = net2sbm(G, label=att, verbose=verbose)
    return {'graph':G, 'sbm_infer':models, 'sbm_label':models_c}

def spl_compare(name='euemail', label='dep', levels=[2, 3], netpath='./snap', sbmpath='./snap/out', **kwargs):
    network = import_network(name=name, dirpath=netpath, directed=False, to_undirected=True, modmax=True)
    net_spl, sbm_spl = {}, {}
    if label:
        assert(label!='modularity')
        try:
            homsbm = net2sbm(network, label=label)
            net_spl[label] = net2spl(network, label=label, k=len(homsbm['sbm']))
            sbm_spl[label] = sbm2emp(homsbm)
        except: pass
    modsbm = net2sbm(network, label='com')
    net_spl['modularity'] = net2spl(network, label='com', k=len(modsbm['sbm']))
    sbm_spl['modularity'] = sbm2emp(modsbm)
    netsbm = import_netsbm(name=name, dirpath=sbmpath, **kwargs)
    blocks = set(list(netsbm['sbm_label'].keys()))
    if levels: blocks = blocks-set([b for b in blocks if not any([str(l) in b for l in levels])])
    for b in blocks:
        net_spl[b] = net2spl(netsbm['graph'], label=b, k=len(netsbm['sbm_label'][b]['sbm']))
        sbm_spl[b] = sbm2emp(netsbm['sbm_label'][b])
    return {'net': net_spl, 'sbm': sbm_spl}

def net2spl(G, label=False, k=None):
    spl = nx.shortest_path_length(G)
    gcc = sorted(nx.connected_components(G), key=len, reverse=True)[0]
    if label:
        labels = {x[0]: x[1][label] for x in G.nodes(data=True)}
        if k is None: k = max(labels.values())+1
        spl_blk = [[[] for j in range(k)] for i in range(k)]
        for ego, alters in spl:
            if ego in gcc:
                blk_ego = labels[ego]
                for alter, d in alters.items():
                    if alter!=ego: spl_blk[blk_ego][labels[alter]].append(d)
        spl_mu, spl_sigma = np.zeros((k,k)), np.zeros((k,k))
        for i in range(k):
            for j in range(k):
                tmp = np.array(spl_blk[i][j])
                spl_mu[i][j], spl_sigma[i][j] = np.nanmean(tmp), np.nanstd(tmp)
        return {'mu': spl_mu, 'sigma': spl_sigma, 'samples':spl_blk}
    else: return dict(spl)

def net2sbm(G, label, verbose=True):
    labels = {x[0]: x[1][label] for x in G.nodes(data=True)}
    labarr = np.array(list(labels.values()))
    adj = nx.to_numpy_array(G, nodelist=labels.keys())
    k = labarr.max()+1
    counts = np.array([[adj[labarr==i,:][:,labarr==j].sum() for j in range(k)] for i in range(k)])
    pi = np.array([(labarr==i).sum() for i in range(k)])
    psi = counts/(np.dot(pi[:,np.newaxis], pi[np.newaxis,:])-np.diag(pi))*len(G)
    psi[np.isnan(psi)] = 0.
    pi = pi/pi.sum()
    model = esbm.StochasticBlockModel(psi, pi)
    if verbose: print('%s | k=%i, network mean degree = %.2f, sbm mean degree = %.2f'%(label, k, adj.sum(0).mean(), model.meanomega))
    spdist = spd(model.get_psi(), len(adj), model.get_pi())
    return {'sbm': model, 'spd': spdist}

def sbm2spl(sbm, n=None, samples=10):
    mus, sigmas = [], []
    k = len(sbm['sbm'])
    if n is None: n = sbm['spd'].num
    for i in range(samples):
        net = sbm['sbm'].generate_networkdata(n=n, networkx_label='block')
        tmp = net2spl(net, 'block', k=k)
        mus.append(tmp['mu'])
        sigmas.append(tmp['sigma'])
    mus = np.stack(mus, axis=0)
    sigmas = np.stack(sigmas, axis=0)
    return {'mu': np.nanmean(mus, 0), 'sigma': np.nanmean(sigmas, 0), 'mu_err': np.nanstd(mus, 0), 'sigma_err': np.nanstd(sigmas, 0)}

def sbm2emp(sbm, samples=10):
    empirical = sbm2spl(sbm, samples=samples)
    analytic = dict(zip(['mu', 'sigma'], sbm['spd'].stats()))
    analytic['sigma'] = np.sqrt(analytic['sigma']) # variance to std dev
    return empirical, analytic

def plot_spl(empirical, analytic, sigma=False, marker='o', color='k', ecolor='red', size=5, alpha=0.25, text=False, 
    label='\\lambda_{ij}', label_x='Empirical', label_y='Analytic', title='', dpi=None, figsize=1.5, ax=None):
    no_axis = ax is None
    ax = get_axes(ax=ax, figsize=(figsize, figsize), dpi=dpi, equal=True, xlabel=label_x, ylabel=label_y, title=title, xscale='int', yscale='int')
    n = empirical['mu'].shape[0]
    x, y = empirical['mu'], analytic['mu']
    if sigma: ax.errorbar(x.flatten(), y.flatten(), xerr=empirical['sigma'].flatten(), yerr=analytic['sigma'].flatten(), marker=marker, alpha=alpha, ms=size, color=color, ecolor=ecolor, ls='')
    else: ax.scatter(x.flatten(), y.flatten(), marker=marker, alpha=alpha, s=size, c=color)
    if text:
        for i in range(n):
            for j in range(i, n): ax.text(x[i][j], y[i][j], '%i-%i'%(i+1,j+1))
    draw_xy(ax)
    if no_axis: return ax

class RandomGraph():

    def __init__(self, mu=None, nu=None, n=2**9, x=None, k=None, k_eff=None, lmax=30, num=2**10, apx=True, tol=1e-10, set_rho=True, set_omega=True, name='random_graph', **kwargs):
        self.set_mu(mu)
        self.set_nu(nu)
        self.set_x(x, k, k_eff)
        self.n = self.get_n(n)
        self.lmax = self.get_n(lmax)
        self.num = self.get_n(num**(1/self.k_eff))
        self.apx = bool(apx)
        assert(tol<=1e-4)
        self.tol = float(tol)
        if set_rho:
            try: self.set_rho()
            except: warn('apt function "set_rho" to compute percolation probabilities not found')
        if set_omega:
            try: self.set_omega(**kwargs)
            except: warn('apt function "set_omega" to compute geodesic lengths not found')
        self.name = str(name)

    def set_x(self, x=None, k=None, k_eff=None):
        def check_keff(k_eff, k):
            if k_eff is not None: k_eff = int(k_eff)
            else: k_eff = k
            assert(0<=k_eff<=k)
            return k_eff
        if x is None:
            try: x = self._mu.support()
            except AttributeError:
                try: x = (self._mu.a, self._mu.b)
                except AttributeError:
                    k = int(k)
                    assert(k>0)
                    self.x = x
                    self.k = k
                    self.k_eff = check_keff(k_eff, k)
                    return
        if isiterable_till(x, 1): x = (x,)
        if isiterable_till(x, 2):
            for i in x:
                if len(i)!=2 or not isordered(i): raise ValueError('expected "x" to be an iterable of 2-tuples of ordered numbers indicating distribution support, but got "%s" instead'%str(x))
            x = tuple([tuple(i) for i in x])
            if k is None: k = len(x)
        else: raise ValueError('expected "x" to be an iterable of 2-tuples of ordered numbers indicating distribution support, but got "%s" instead'%str(x))
        self.x = x
        k = int(k)
        assert(k>0)
        if k!=len(self.x): warn('"k"=%i does not match dimensionality of given space "x" (%i)'%(k, len(self.x)))
        self.k = k
        self.k_eff = check_keff(k_eff, k)

    def subset_x(self, x=None, check_inf=False):
        if x is None: x = self.x
        if isiterable_till(x, 1): x = (x,)*self.k
        if isiterable_till(x, 2):
            if len(x)!=self.k: raise ValueError('expected "x" to have length %i, but got length %i instead'%(self.k, len(x)))
            for i in range(self.k):
                if len(x[i])!=2 or not self.x[i][0]<=x[i][0]<=x[i][1]<=self.x[i][1]: raise ValueError('expected "x" to be an iterable of 2-tuples of ordered numbers indicating subset of distribution support, but got "%s" instead'%str(x))
                if check_inf and np.isinf(x[i]).any(): raise ValueError('expected "x" to contain finite values only')
            return tuple([tuple(i) for i in x])
        else: raise ValueError('expected "x" to be an iterable of 2-tuples of ordered numbers indicating subset of distribution support, but got "%s" instead'%str(x))

    def set_mu(self, mu=None):
        if mu is None:
            from scipy.stats import uniform
            mu = uniform()
        assert(hasattr(mu, 'pdf'))
        self._mu = mu

    def set_nu(self, nu=None):
        if nu is None: nu = 2.
        if isinstance(nu, (float, int)): self._nu = lambda x, y: float(nu)
        else:
            assert(callable(nu))
            self._nu = nu

    def get_n(self, n=None):
        if n is None: return self.n
        n = int(n)
        assert(n>0)
        return n

    def get_l(self, l=None):
        if l is None: l = self.lmax
        l = int(l)
        assert(l>0)
        return list(range(l+1))

    def mu(self, x): return self._mu.pdf(x)

    def expect(self, foo): return self._mu.expect(foo)

    def mean_location(self):
        if hasattr(self._mu, 'mean'):
            if callable(self._mu.mean): return self._mu.mean()
            else: return self._mu.mean
        else: raise NotImplementedError('"mean" attribute or method not implemented for given "mu"')

    def logmu(self, x): return self._mu.logpdf(x)

    def nu(self, x, y, n=None): return self._nu(x, y)/self.get_n(n)

    def lognu(self, x, y, n=None): return np.log(self._nu(x, y))-np.log(self.get_n(n))

    def degree(self, x=None):
        if x is None: return self.expect(lambda x: self.degree(x))
        elif isiterable(x): return [self.degree(i) for i in x]
        else: return self.expect(lambda y: self._nu(x, y))

    def transform(self, x): #for 1-d models, this must not be updated or else some functionalities may not work!
        try: return self._mu.cdf(x)
        except AttributeError: raise NotImplementedError('transforming coordinates not yet implemented')

    def transform_inv(self, x): #for 1-d models, this must not be updated or else some functionalities may not work!
        try: return self._mu.ppf(x)
        except AttributeError: raise NotImplementedError('inverse transforming coordinates not yet implemented')

    def get_nodes(self, space=None, transform=False, grid=True, check_inf=True):
        x = self.subset_x(space)
        if transform:
            x = self.transform(x)
            x = [np.linspace(x[i][0], x[i][1], self.num) for i in range(self.k)]
            x = self.transform_inv(x)
            if check_inf: x = [i[~np.isinf(i)] for i in x]
        elif check_inf and np.isinf(x).any(): raise ValueError('expected a finite subset to get node locations from, but got %s instead'%str(x))
        else: x = [np.linspace(x[i][0], x[i][1], self.num) for i in range(self.k)]
        if self.k>1 and grid:
            from itertools import product
            x = list(product(*x))
        if self.k==1: x = x[0]
        return np.array(x)

    def get_ax(self, ax=None, axis_frame=True, axis_x=True, axis_y=True, equal=False, **kwargs):
        from matplotlib.pyplot import subplots
        if ax is None: fig, ax = subplots(**kwargs)
        if equal: ax.set_aspect('equal')
        if not axis_frame: ax.set_frame_on(False)
        if not axis_x: ax.set_xticks([])
        if not axis_y: ax.set_yticks([])
        return ax

    def get_interpolator(self, x, y, z=None, **kwargs):
        from scipy import interpolate
        if z is None:
            return interpolate.interp1d(x, y, kind='cubic', **kwargs)
        else:
            points, value = np.vstack([x.flatten(), y.flatten()]).T, z.flatten()
            return interpolate.CloughTocher2DInterpolator(points, value)

    def from_interpolator(self, x, y=None, obj=None, **kwargs):
        if y is None: return obj(x)
        else: return obj(x, y)

    def plot_curve(self, x, y, ax=None, ax_kwargs=dict(axis_frame=False, axis_x=False, axis_y=True, equal=False), color='k', **kwargs):
        no_axis = ax is None
        ax = self.get_ax(ax=ax, **ax_kwargs)
        ax.plot(x, y, c=color, **kwargs)
        if no_axis: return ax

    def plot_contour(self, x, y, z, ax=None, ax_kwargs=dict(axis_frame=False, axis_x=False, axis_y=False, equal=True), lines=False, label_lines=True, tri=False, cmap='Blues', cbar=True, color='k', fontsize=10, levels=10, line_kwargs=dict(), **kwargs):
        no_axis = ax is None
        ax = self.get_ax(ax=ax, **ax_kwargs)
        if tri:
            col = ax.tricontourf(x, y, z, cmap=cmap, levels=levels, **kwargs)
            if lines: tmp = ax.tricontour(x, y, z, colors=color, levels=list(range(int(np.ceil(z.min())), int(np.ceil(z.max())))), **line_kwargs)
        else:
            col = ax.contourf(x, y, z, cmap=cmap, levels=levels, **kwargs)
            if lines: tmp = ax.contour(x, y, z, colors=color, levels=list(range(int(np.ceil(z.min())), int(np.ceil(z.max())))), **line_kwargs)
        if lines and label_lines: ax.clabel(tmp, inline=True, fontsize=fontsize, fmt='%i', inline_spacing=2)
        if cbar: ax.get_figure().colorbar(col, ax=ax, fraction=0.1, pad=0.02, format='%.2f')
        if no_axis: return ax

    def plot_point(self, x, y, ax=None, ax_kwargs=dict(axis_frame=False, axis_x=False, axis_y=False, equal=True), color='k', marker='x', s=2, xlim=None, ylim=None, **kwargs):
        no_axis = ax is None
        ax = self.get_ax(ax=ax, **ax_kwargs)
        ax.scatter(x, y, color=color, marker=marker, s=s, **kwargs)
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)
        if no_axis: return ax

    def plot_bar(self, x, y, xerr=None, yerr=None, ax=None, ax_kwargs=dict(axis_frame=False, axis_x=False, axis_y=False, equal=False), color='k', marker='$\\circ$', s=None, ls='', thickness=None, **kwargs):
        no_axis = ax is None
        ax = self.get_ax(ax=ax, **ax_kwargs)
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, color=color, ls=ls, marker=marker, ms=s, elinewidth=thickness, **kwargs)
        if no_axis: return ax

    def plot_line(self, a, b, ax=None, ax_kwargs=dict(axis_frame=False, axis_x=False, axis_y=False, equal=True), linewidth=0.2, linestyle='-', color='k', marker=None, markercolor='k', markersize=1, pad=0.05, xlim=None, ylim=None, **kwargs):
        no_axis = ax is None
        ax = self.get_ax(ax=ax, **ax_kwargs)
        from matplotlib.lines import Line2D
        if isiterable_till(a, 1) and isiterable_till(b, 1): a, b = [a], [b]
        if isiterable_till(a, 2) and isiterable_till(b, 2):
            a, b = np.array(a), np.array(b)
            for i in range(a.shape[0]): ax.add_line(Line2D((a[i,0], b[i,0]), (a[i,1], b[i,1]), linewidth=linewidth, linestyle=linestyle, color=color, marker=marker, markeredgecolor=markercolor, markerfacecolor=markercolor, markersize=markersize, **kwargs))
            xlim_curr, ylim_curr = ax.get_xlim(), ax.get_ylim()
            vmin, vmax = np.vstack([a.min(axis=0), b.min(axis=0)]).min(axis=0), np.vstack([a.max(axis=0), b.max(axis=0)]).max(axis=0)
            delta = pad*(vmax-vmin)/2
            vmax += delta
            vmin -= delta
            if xlim is None: ax.set_xlim(min(xlim_curr[0], vmin[0]), max(xlim_curr[1], vmax[0]))
            else: ax.set_xlim(xlim)
            if ylim is None: ax.set_ylim(min(ylim_curr[0], vmin[1]), max(ylim_curr[1], vmax[1]))
            else: ax.set_ylim(ylim)
        else: raise ValueError('expected "a" and "b" to indicate locations to place lines between')
        if no_axis: return ax

    def plot_vline(self, x, ax=None, ax_kwargs=dict(axis_frame=False, axis_x=False, axis_y=False, equal=False), linewidth=None, linestyle='--', color='tab:red', **kwargs):
        no_axis = ax is None
        if not isiterable(x): x = [x]
        ax = self.get_ax(ax=ax, **ax_kwargs)
        for i in x: ax.axvline(i, linewidth=linewidth, linestyle=linestyle, color=color, **kwargs)
        if no_axis: return ax

    def plot_hline(self, y, ax=None, ax_kwargs=dict(axis_frame=False, axis_x=False, axis_y=False, equal=False), linewidth=None, linestyle='--', color='tab:red', **kwargs):
        no_axis = ax is None
        if not isiterable(y): y = [y]
        ax = self.get_ax(ax=ax, **ax_kwargs)
        for i in y: ax.axhline(i, linewidth=linewidth, linestyle=linestyle, color=color, **kwargs)
        if no_axis: return ax

    def plot_edges(self, G=None, transform=False, space=None, node_kwargs=dict(), ax=None, gcc_color=None, scc_color='k', **kwargs):
        if G is None: G = self.sample()
        n = len(G.nodes)
        x = [G.nodes[i]['x'] for i in range(n)]
        if transform: x = self.transform(x)
        a, b = [], []
        for i in range(n):
            for j in G.adj[i]:
                a.append(x[i])
                b.append(x[j])
        a, b = np.array(a), np.array(b)
        if space is not None and a.ndim==2: xlim, ylim = self.subset_x(space)
        else: xlim, ylim = None, None
        if a.ndim==2 and a.shape[1]==1: a, b = a[:,0], b[:,0]
        if a.ndim==1: return self.plot_point(a, b, marker='s', ax=ax, **kwargs)
        elif a.ndim==2:
            axnone = ax is None
            if axnone:
                ax = self.plot_line(a, b, xlim=xlim, ylim=ylim, **kwargs)
            else:
                self.plot_line(a, b, xlim=xlim, ylim=ylim, ax=ax, **kwargs)
            x = np.array(x)
            if gcc_color:
                gcc = self.get_components(G, largest=True)
                color = [gcc_color if i in gcc else scc_color for i in range(n)]
            else:
                color = scc_color
            self.plot_point(x[:, 0], x[:, 1], xlim=xlim, ylim=ylim, ax=ax, color=color, **node_kwargs)
            return ax
        else: raise NotImplementedError('cannot plot edges in this nodespace')

    def plot_mean(self, transform=False, vertical=True, s=10, color='tab:red', **kwargs):
        x = self.mean_location()
        if transform: x = self.transform(x)
        if not isiterable(x) or len(x)==1:
            if vertical: self.plot_vline(x, color=color, **kwargs)
            else: self.plot_hline(x, color=color, **kwargs)
        elif len(x)==2: self.plot_point(x[0], x[1], s=s, color=color, **kwargs)
        else: raise NotImplementedError('cannot plot mean in this nodespace')

    def plot_nodespace(self, space=None, transform=False, foo=None, foo_kwargs=dict(), **kwargs):
        x = self.get_nodes(space, transform=transform)
        if foo is None: foo = self.mu
        y = foo(x, **foo_kwargs)
        if self.k_eff==1: return self.plot_curve(x, y, **kwargs)
        elif self.k_eff==2: return self.plot_contour(*self.get_nodes(space, transform=transform, grid=False), y.reshape((int(np.sqrt(len(x))), -1)).T, **kwargs)
        else: raise NotImplementedError('cannot plot in nodespace when ndims>2')

    def plot_edgespace(self, space=None, transform=False, foo=None, foo_kwargs=dict(), **kwargs):
        x = self.get_nodes(space, transform=transform)
        if foo is None:
            foo = self.nu
            if 'n' not in foo_kwargs:
                foo_kwargs = foo_kwargs.copy()
                foo_kwargs['n'] = 1
        if self.k_eff==1:
            y = np.array([[foo(xx, yy, **foo_kwargs) for yy in x] for xx in x]).T
            return self.plot_contour(x, x, y, **kwargs)
        elif self.k_eff==2:
            try:
                yy = self.mean_location()
                y = np.array([foo(xx, yy, **foo_kwargs) for xx in x]).reshape((int(np.sqrt(len(x))), -1)).T
                return self.plot_contour(*self.get_nodes(space, transform=transform, grid=False), y, **kwargs)
            except AttributeError: raise NotImplementedError('cannot plot in edgespace when ndims>1')
        else: raise NotImplementedError('cannot plot in nodespace when ndims>2')

    def plot(self, G=None, data=None, samples=None, bins=20, title_loc='top', space=None, transform=False, dpi=180, figsize=2, cbar_pad=1.2, **kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=2, ncols=3, dpi=dpi, figsize=(figsize*3*cbar_pad, figsize*2), constrained_layout=True)
        self.plot_nodespace(space=space, transform=transform, ax=ax[0,0])
        self.plot_edgespace(space=space, transform=transform, ax=ax[0,1])
        self.plot_edges(space=space, G=G, transform=transform, ax=ax[0,2], **kwargs)
        self.plot_nodespace(space=space, transform=transform, foo=self.degree, ax=ax[1,0])
        self.plot_nodespace(space=space, transform=transform, foo=self.rho, ax=ax[1,1])
        self.plot_edgespace(space=space, transform=transform, foo=self.spl, ax=ax[1,2])
        for i in range(2):
            for j in range(3):
                if not (i==0 and j==2): self.plot_mean(transform=transform, ax=ax[i,j], ax_kwargs=dict())
        if self.k_eff==1: tgt_lab = 'y'
        else: tgt_lab = '\\langle x\\rangle'
        draw_text(ax[0,0], 'Density $\\mu(x)$', title_loc)
        draw_text(ax[0,1], 'Kernel $\\nu(x, %s)$'%tgt_lab, title_loc)
        draw_text(ax[0,2], 'Sample $(n=%i)$'%self.n, title_loc)
        draw_text(ax[1,0], 'Degree $d(x)$', title_loc)
        draw_text(ax[1,1], 'Percolation $\\rho(x)$', title_loc)
        draw_text(ax[1,2], 'Geodesic $\\langle\\lambda_{x%s}\\rangle$'%tgt_lab, title_loc)
        return ax

    def set_rho(self):
        if self.k_eff>1: raise NotImplementedError('interpolation based kernel integral not implemented for k_eff>1')
        warn('setting interpolation based kernel integral to compute percolation probabilities may take some time; try changing "num"')
        xs = self.get_nodes(transform=True)
        num = len(xs)
        kernel = np.array([[self._nu(x, y) for y in xs] for x in xs])
        def foo(rho=0.): return 1 - np.exp(-kernel@rho/num) #consistency equation for size of GCC
        rho_old, rho_new = np.ones(num), np.ones(num)
        while True:
            rho_new = foo(rho_old)
            if ((rho_new-rho_old)**2).mean()<self.tol: break
            rho_old = rho_new
        self._rho = self.get_interpolator(xs, rho_new)

    def rho(self, x=None):
        if x is None: return self.expect(lambda x: self.rho(x))
        else: return self.from_interpolator(x, obj=self._rho)

    def is_percolating(self): return not(np.isclose(self.rho(), 0))

    def set_omega(self, **kwargs):
        if self.k_eff>1: raise NotImplementedError('interpolation based kernel integral not implemented for k_eff>1')
        warn('setting interpolation based kernel integral to compute geodesic statistics may take some time; try changing "num"')
        xs = self.get_nodes(transform=True)
        num = len(xs)
        xx, yy = xs[:,None]*np.ones((num, num)), xs[None,:]*np.ones((num, num))
        kernel = np.array([[self._nu(x, y) for y in xs] for x in xs])
        out = [kernel]
        if not self.apx:
            rho = self.rho(xs)
            factor = 1+(1/rho[:,None]-1)*rho[:,None]
            factor[np.isinf(factor)] = 0
            factor[np.isnan(factor)] = 0
            out[-1] = out[-1]*factor
        for i in range(self.lmax-1): out.append(out[-1]@kernel/num)
        self._omega = tuple([0.]+[self.get_interpolator(xx, yy, o) for o in out])
        self._omega_cumsum = tuple([0.]+[self.get_interpolator(xx, yy, o) for o in np.array(out).cumsum(axis=0)])

    def omega(self, x, y, l, n=None):
        if not isinstance(l, int) or l<0: raise ValueError('expected given length to be a non-negative int')
        elif l==0: return 0.
        elif l==1 and self.apx: return self.nu(x, y, n)
        elif l<=self.lmax: return self.from_interpolator(x, y, obj=self._omega[l])/self.get_n(n)
        else: raise ValueError('given length of %i is longer than lmax of %i; set lmax aptly'%(l, self.lmax))

    def psi(self, x, y, l=None, n=None):
        sup = self.get_l(l)
        out = np.exp(-np.array([self.omega(x, y, i, n) for i in sup]).cumsum())
        if l is not None: out = out[l]
        return out

    def spl(self, x, y, n=None): return self.psi(x, y, n=n).sum()

    def aspl(self, x=None, n=None):
        if x is None: return self.expect(lambda x: self.aspl(x, n))
        elif isiterable(x): return [self.aspl(i, n) for i in x]
        else: return self.expect(lambda y: self.spl(x, y, n))

    def closeness(self, x, n=None):
        if isiterable(x): return [self.closeness(i, n) for i in x]
        else: return 1/self.expect(lambda y: self.spl(y, x, n))

    def betweenness(self, x, n=None): raise NotImplementedError('computation of betweenness with apx closed-form spd not yet implemented')

    def dot_long(self, x, y, M):
        assert(x.ndim==y.ndim==1 and M.ndim==3 and M.shape[1]==M.shape[2]==len(x))
        return (x[None,:,None]*M*y[None,None,:]).sum(axis=2).sum(axis=1)

    def sample(self, samples=1, n=None, gcc=False, seed=None):
        if samples>1: return [self.sample(n=n, gcc=gcc) for i in range(int(samples))]
        import networkx as nx
        from scipy.stats import bernoulli
        if seed is not None:
            import numpy as np
            np.random.seed(seed)
        n = self.get_n(n)
        x = self._mu.rvs(size=n)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            G.nodes[i]['x'] = x[i]
            for j in range(i):
                edge = bernoulli.rvs(min(1, self.nu(x[i], x[j], n)))
                if edge: G.add_edge(i, j)
        if gcc:
            comp = self.get_components(G)
            for i in range(1, len(comp)): G.remove_nodes_from(comp[i])
        return G

    def get_components(self, G, largest=False, num=False):
        if isinstance(G, (list, tuple)): return [self.get_components(g) for g in G]
        from networkx import connected_components
        out = sorted(connected_components(G), key=len, reverse=True)
        if largest:
            out = out[0]
        elif num:
            out = len(out)
        return out

    def rho_empirical(self, G):
        if isinstance(G, (list, tuple)): return [self.rho_empirical(g) for g in G]
        return len(self.get_components(G, largest=True))/len(G)

    def spl_empirical(self, G, mean_network=False):
        if isinstance(G, (list, tuple)): return [self.spl_empirical(g, mean_network) for g in G]
        from networkx import shortest_path_length, average_shortest_path_length
        if mean_network: return average_shortest_path_length(G)
        else: return dict(shortest_path_length(G))

    def centrality_empirical(self, G, kind='h'):
        if isinstance(G, (list, tuple)): return [self.centrality_empirical(g, kind) for g in G]
        if kind=='b': from networkx.algorithms.centrality.betweenness import betweenness_centrality as centrality
        elif kind=='c': from networkx.algorithms.centrality.closeness import closeness_centrality as centrality
        elif kind=='h': from networkx.algorithms.centrality.harmonic import harmonic_centrality as centrality
        else: from networkx.algorithms.centrality.degree_alg import degree_centrality as centrality
        return centrality(G)

    def degree_empirical(self, G, mean_network=False):
        if isinstance(G, (list, tuple)): return [self.degree_empirical(g, mean_network) for g in G]
        if mean_network:
            from networkx.classes.function import degree
            return np.mean(degree(G))
        else: 
            from networkx.classes.function import degree_histogram
            dist = np.array(degree_histogram(G))
            return dist/dist.sum()

    def eigenmean_empirical(self, G):
        from networkx import to_numpy_array
        import numpy as np
        adj = to_numpy_array(G, dtype=bool)
        val, vec = np.linalg.eigh(adj)
        idx = np.argmax(np.abs(val))
        val, vec = val[idx], vec[:, idx]
        if val<0: val = -val
        if np.all(vec<0): vec = -vec
        return np.log(val), np.log(vec).mean()

    def groupby(self, G, data, mean=True, label='x', foo=None):
        if isinstance(G, (list, tuple)):
            out = [self.groupby(g, d, mean, label, foo) for (g, d) in zip(G, data)]
            if mean:
                out = [dict(zip(*o)) for o in out]
                out_x = list(set(np.hstack([list(o.keys()) for o in out])))
                out_y = [[o[i] for o in out if i in o] for i in out_x]
                out_y = np.array([(np.mean(i), np.std(i)) for i in out_y])
                return out_x, out_y
        if foo is None: foo = lambda x: x
        x = {i[0]: foo(i[1][label]) for i in G.nodes(data=True)}
        out = {}
        for key in x:
            if key in data:
                if x[key] not in out: out[x[key]] = []
                out[x[key]].append(data[key])
        out_x = list(set(out.keys()))
        out_y = [out[i] for i in out_x]
        if mean: out_y = [np.mean(i) for i in out_y]
        return out_x, out_y

    def rho_compare(self, samples=10, n=None):
        if isinstance(samples, (int, float)): samples = self.sample(samples, n)
        if isinstance(samples, (list, tuple)): return [self.rho_compare(samples=G, n=n) for G in samples]
        return self.rho_empirical(samples), self.rho()

    def spl_compare(self, samples=10, n=None, gcc=True, mean_node=False, mean_network=False):
        if isinstance(samples, (int, float)): samples = self.sample(samples, n, gcc)
        if isinstance(samples, (list, tuple)): return [self.spl_compare(samples=G, n=n, gcc=gcc, mean_node=mean_node, mean_network=mean_network) for G in samples]
        data = []
        x = {i[0]: i[1]['x'] for i in samples.nodes(data=True)}
        if mean_node:
            if mean_network: raise ValueError('only one of "mean_node" or "mean_network" can be True')
            lens = self.spl_empirical(samples, mean_network=False)
            for i in lens: data.append([np.mean(lens[i].values()), self.aspl(x[i], n=n)])
        elif mean_network: 
            data = self.spl_empirical(samples, mean_network=True)
            ana = self.aspl(n=n)
            if isinstance(data, (list, tuple)): data = [(d, ana) for d in data]
            else: data = [data, ana]
        else:
            lens = self.spl_empirical(samples, mean_network=False)
            for i in lens:
                for j in lens[i]: data.append([lens[i][j], self.spl(x[i], x[j], n=n)])
        data = np.array(data)
        return data

    def centrality_compare(self, kind='h', samples=10, n=None, gcc=True):
        if isinstance(samples, (int, float)): samples = self.sample(samples, n, gcc)
        if isinstance(samples, (list, tuple)): return [self.centrality_compare(samples=G, n=n, gcc=gcc) for G in samples]
        if kind=='b': foo = self.betweenness
        elif kind=='c': foo = self.closeness
        elif kind=='h': foo = self.closeness
        else: foo = self.degree
        data = []
        x = {i[0]: i[1]['x'] for i in samples.nodes(data=True)}
        cent = self.centrality_empirical(samples, kind)
        for i in cent: data.append([cent[i], foo(x[i], n=n)])
        data = np.array(data)
        return data

    def eigenmean_compare(self, samples=10, n=None, gcc=True):
        if isinstance(samples, (int, float)): samples = self.sample(samples, n, gcc)
        ana = self.to_rankone(n=n).eigenmean_analytic()
        if isinstance(samples, (list, tuple)): return np.dstack([np.vstack([self.eigenmean_empirical(G), ana]).T for G in samples])
        else: return np.vstack([self.eigenmean_empirical(samples), ana]).T

    def plot_deg(self, G=None, n=None, ax=None, dpi=None, xlabel='Degree $d$', ylabel='$P(d)$', title='', **kwargs):
        if G is None: G = self.sample(n=n)
        dist = self.degree_empirical(G)[1:]
        degs = np.arange(len(dist))+1
        idx = dist!=0
        degs, dist = degs[idx], dist[idx]
        no_axis = ax is None
        ax = get_axes(ax=ax, figsize=(4, 3), dpi=dpi, xlabel=xlabel, ylabel=ylabel, title=title)
        ax.loglog(degs, dist, ls='', **kwargs)
        if no_axis: return ax

    def plot_spl(self, data=None, stddev=False, bins=20, samples=10, n=None, ax=None, dpi=None, xlabel='Approximate Closed-form', ylabel='Empirical', title='', marker='$\\circ$', color='k', size=5, thickness=None):
        if data is None: data = self.spl_compare(samples, n)
        if not isinstance(data, (list, tuple)): data = [data]
        vmin = min([d[:,1].min() for d in data])
        vmax = min([d[:,1].max() for d in data])
        partition = np.linspace(vmin, vmax, bins+1)
        idx = [np.digitize(d[:,1], partition) for d in data]
        for i in range(len(idx)): idx[i][idx[i]>bins] = bins
        if stddev:
            mus, sigmas = [], []
            for i in range(1, bins+1):
                mu, sigma = [], []
                for j in range(len(idx)):
                    d = data[j][idx[j]==i]
                    mu.append(d.mean(axis=0))
                    sigma.append(d.std(axis=0))
                mus.append(mu)
                sigmas.append(sigma)
            mu, sigma = np.array(mus).mean(axis=1), np.array(sigmas).mean(axis=1)
        else:
            outs = []
            for i in range(1, bins+1):
                out = []
                for j in range(len(idx)): out.append(data[j][idx[j]==i].mean(axis=0))
                outs.append(out)
            outs = np.array(outs)
            mu, sigma = outs.mean(axis=1), outs.std(axis=1)
        no_axis = ax is None
        ax = get_axes(ax=ax, figsize=(3, 3), dpi=dpi, equal=True, xscale='int', yscale='int', xlabel=xlabel, ylabel=ylabel, title=title)
        ax.errorbar(mu[:,1], mu[:,0], xerr=sigma[:,1], yerr=sigma[:,0], color=color, ls='', marker=marker, ms=size, elinewidth=thickness)
        draw_xy(ax)
        if no_axis: return ax

    def plot_spl_all(self, data, title=[], subtitle=[], stddev=False, bins=20, dpi=180, figsize=2, textpad=0.05, labelpad=20, hpad=0.9, subtitleloc='right', xlabel='Approximate Closed-form', ylabel='Empirical', **kwargs):
        if not title: title = ['']*len(data)
        if not subtitle: subtitle = ['']*len(data)
        import matplotlib.pyplot as plt
        naxes = len(data)
        ncols = min(naxes, 3)
        nrows = naxes//3 + int(naxes%3!=0)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize*ncols*hpad, figsize*nrows), dpi=dpi, squeeze=False)
        idx = 0
        for i in range(nrows):
            for j in range(ncols):                
                if idx<naxes:
                    self.plot_spl(data=data[idx], stddev=stddev, bins=bins, ax=ax[i,j], title=title[idx], xlabel='', ylabel='', **kwargs)
                    if subtitle[idx]: draw_text(ax[i,j], subtitle[idx], subtitleloc, textpad)
                else: ax[i,j].set_axis_off()
                idx += 1
        ax_outer = fig.add_subplot(frame_on=False, xticks=[], yticks=[])
        if xlabel: ax_outer.set_xlabel(xlabel, labelpad=labelpad)
        if ylabel: ax_outer.set_ylabel(ylabel, labelpad=labelpad)
        plt.tight_layout()
        return ax

    def to_sbm(self, partition=10, n=None):
        import scipy.integrate as integrate
        if self.k>1: raise NotImplementedError('conversion to SBM for k>1 not yet implemented')
        if isinstance(partition, (float, int)): partition = self._mu.ppf(np.linspace(0, 1, partition+1))
        self.subset_x((min(partition), max(partition)))
        p = np.diff(self._mu.cdf(partition))
        k = len(p)
        if not (p>0).all(): raise ValueError('expected an ordered partition of the node (sub)space')
        B = np.array([[integrate.dblquad(lambda a, b: self._nu(a, b)*self.mu(a)*self.mu(b), partition[i], partition[i+1], lambda a:partition[j], lambda a:partition[j+1])[0]/(p[i]*p[j]) for j in range(k)] for i in range(k)])
        B = (B+B.T)/2 #only support undirected SBMs for now
        return SBM(B=B, pi=p, name=self.name+'_sbm', n=self.get_n(n))

    def critical_empirical(self, param, samples=1, n=1000): return np.array(self.foo_size_empirical(*self.critical_xy(), self.critical_z(param), samples=samples, n=n))

    def critical_analytic(self, param, boolean=False): return np.array(self.foo_size_analytic(*self.critical_xy(), self.critical_z(param), boolean=boolean))

    def plot_critical(self, x=None, y=None, z=None, smooth=0, samples=False, rho=False, n=1000, num=1000, levels=np.linspace(0,1,11), ax=None, ax_kwargs=dict(), clabel='', zlabel='$\\langle\\rho\\rangle$', color='red', cmap='Blues', **kwargs):
        if x is None: x = self.critical_xy()[0]
        if y is None: y = self.critical_xy()[1]
        x_fine, y_fine = np.linspace(x.min(), x.max(), num), np.linspace(y.min(), y.max(), num)
        no_axis = ax is None
        ax = get_axes(ax=ax, **ax_kwargs)
        if z is None:
            if samples:
                try:
                    if isinstance(samples, (float, int)): z = self.foo_size_empirical(x, y, samples=samples, n=n, **kwargs)
                    z = np.array(z)
                    if z.ndim==3: z = z.mean(axis=-1)
                except: pass
            elif rho:
                try: z = self.foo_size_analytic(x, y, **kwargs)
                except: pass
        if z is not None:
            if smooth:
                from scipy.ndimage.filters import gaussian_filter
                z = gaussian_filter(z, sigma=smooth)
            c = ax.contourf(x, y, np.array(z).T, cmap=cmap, levels=levels)
            ax.get_figure().colorbar(c, ax=ax, label=zlabel)
        cs = ax.contour(x_fine, y_fine, np.array(self.foo_critical(x_fine, y_fine, **kwargs)).T, [0], linestyles='solid', colors=color)
        if clabel:
            txt = ax.clabel(cs, inline=True, fontsize='smaller')
            txt[0].set_text(clabel)
        if no_axis: return ax

    def plot_critical_z(self, x, y, z, ax=None, ax_kwargs=dict(), c_kwargs=dict(), **kwargs):
        no_axis = ax is None
        ax = get_axes(ax=ax, **ax_kwargs)
        if not isiterable(z): z = [z]
        colors = get_color(z, ax=ax, **c_kwargs)
        for i in range(len(z)): ax.contour(x, y, np.array(self.foo_critical(x, y, z[i])).T, [0], colors=[colors[i]], **kwargs)
        if no_axis: return ax

class SBM(RandomGraph):

    def __init__(self, B=[[1, 4], [4, 1]], pi=[0.4, 0.6], name='sbm', **kwargs):
        self.set_params(B, pi)
        RandomGraph.__init__(self, mu=None, nu=lambda x, y: self.B[self.get_block(x),self.get_block(y)], name=name, **kwargs)

    def set_params(self, B=[[1, 4],[4, 1]], pi=[0.4, 0.6]):
        B = np.array(B)
        pi = np.array(pi)
        assert(B.ndim==2 and pi.ndim==1 and len(pi)==B.shape[0]==B.shape[1])
        assert(np.all(B>=0))
        assert(np.all(B==B.T))
        pi = pi/pi.sum()
        assert(np.all(pi>0))
        self.B = B
        self.pi = pi
        self._k = len(self.pi)
        self._Pi = np.diag(self.pi)
        self._pi_cumsum = np.hstack([[0.], pi.cumsum()])
        self._pi_cumsum[-1] += 1e-8 #to cover edge case of x=1

    def set_helpers(self):
        #may take a while, only call when rank-one conversion or shortest-path based statistics are needed
        pi_sqrt = np.sqrt(self.pi)
        eigval, eigvec = np.linalg.eigh(pi_sqrt[:,None]*self.B*pi_sqrt[None,:]) #for conversion to rank-one graph
        eigvec = (1/pi_sqrt)[:,None]*eigvec
        idx = np.argsort(eigval)[::-1]
        self._rank1_beta, self._rank1_foo = np.sqrt(eigval[idx[0]]), eigvec[:, idx[0]]
        if np.all(self._rank1_foo<0): self._rank1_foo = -self._rank1_foo #eigvec can mistakenly be all negative
        self._spd = self.to_spd()
        self._spd_closeness = self._spd.closeness()
        self._spd_betweenness = self._spd.betweenness()
        self._spd_betweenness_apx = self._spd.betweenness(independent=True)

    def get_block(self, x): return np.digitize(x, self._pi_cumsum)-1

    def get_x(self, blocks=None):
        #creates representative location points
        if blocks is None: blocks = np.arange(self._k)
        flag = False
        if isinstance(blocks, (float, int)):
            flag = True
            blocks = [blocks]
        x = list(self._pi_cumsum)
        x[-1] = 1 #rectify edge case
        x = np.array(x[:-1])+np.diff(x)/2
        x = [x[b] for b in blocks]
        if flag: x = x[0]
        return x

    def expect(self, foo): return np.array([foo(i) for i in self.get_x()])@self.pi

    def plot_blockspace(self, data=None, blocks=None, foo=None, foo_kwargs=dict(), curve_kwargs=dict(), **kwargs):
        if blocks is None: blocks = np.arange(self._k)
        x = self.get_x(blocks)
        if data is not None:
            data = np.array(data)
            yerr = None
            if data.shape[0]!=len(x): raise ValueError('expected data for %i block-points'%len(x))
            if data.ndim==1: y = data
            elif data.ndim==2:
                if data.shape[1]==1: y = data[:,0]
                elif data.shape[1]==2:
                    y, yerr = data[:,0], data[:,1]
                else: raise ValueError('expected data to be a length-%i vector or size-(%i x 2) matrix'%(len(x), len(x)))
            else: raise ValueError('expected data to be a length-%i vector or size-(%i x 2) matrix'%(len(x), len(x)))
            if foo is None: return self.plot_bar(x, y, yerr=yerr, **kwargs)
            else:
                y_s = np.array([foo(xx, **foo_kwargs) for xx in x])
                if 'ax' in kwargs:
                    ax = kwargs['ax']
                    self.plot_curve(x, y_s, ax=ax, **curve_kwargs)
                else: ax = self.plot_curve(x, y_s, **curve_kwargs)
                pi = np.array([self.pi[b] for b in blocks])
                scale_to = (y_s*pi).sum()/pi.sum()
                scale_fr = (y*pi).sum()/pi.sum()
                y = y*scale_to/scale_fr
                if yerr is not None: yerr = yerr*scale_to/scale_fr
                if 'ax' in kwargs: self.plot_bar(x, y, yerr=yerr, **kwargs)
                else: self.plot_bar(x, y, yerr=yerr, ax=ax, **kwargs)
                if 'ax' not in kwargs: return ax
        elif foo is not None: return self.plot_curve(x, np.array([foo(xx, **foo_kwargs) for xx in x]), **curve_kwargs, **kwargs)

    def degree(self, x=None):
        if x is None: return self.pi@self.B@self.pi
        else:
            if isiterable(x): return [(self.B@self.pi)[i] for i in self.get_block(x)]
            else: return (self.B@self.pi)[self.get_block(x)]

    def closeness(self, x, n=None, exact=True):
        if isiterable(x): return [self.closeness(i, n, exact) for i in x]
        else: 
            if exact:
                if self.get_n(n)==self.n: c = self._spd_closeness
                else: c = self.to_spd(n).closeness()
                return c[self.get_block(x)]
            else: return super().closeness(x, n)

    def betweenness(self, x, n=None, exact=True):
        if isiterable(x): return [self.betweenness(i, n, exact) for i in x]
        else: 
            if exact:
                if self.get_n(n)==self.n: c = self._spd_betweenness
                else: c = self.to_spd(n).betweenness()
                return c[self.get_block(x)]
            else:
                try:
                    if self.get_n(n)==self.n: c = self._spd_betweenness_apx
                    else: c = self.to_spd(n).betweenness(independent=True)
                    return c[self.get_block(x)]
                except: return super().betweenness(x, n)

    def set_rho(self):
        def foo(rho=0.): return 1 - np.exp(-self.B@(rho*self.pi)) #consistency equation for size of GCC
        rho_old, rho_new = np.ones(self._k), np.ones(self._k)
        while True:
            rho_new = foo(rho_old)
            if ((rho_new-rho_old)**2).mean()<self.tol: break
            rho_old = rho_new
        self._rho = rho_new

    def rho(self, x=None):
        if x is None: return self.pi@self._rho
        else:
            if isiterable(x): return [self._rho[i] for i in self.get_block(x)]
            else: return self._rho[self.get_block(x)]

    def is_percolating(self): return not(np.allclose(self._rho, 0))

    def set_omega(self, set_helpers=False, **kwargs):
        #for faster computation of psi and average spl
        out = [np.zeros((self._k, self._k)), self.B]
        for i in range(self.lmax-1): out.append(out[-1]@self._Pi@self.B)
        self._omega = np.array(out)
        self._omega_cumsum = self._omega.cumsum(axis=0)    
        if set_helpers:    
            self.set_helpers()

    def get_omega(self, l=None):
        if l is None: return self._omega
        elif 0<=l<=self.lmax: return self._omega[l]
        else: raise ValueError('expected "l" to be between 0 and lmax (%i)'%self.lmax)

    def omega(self, x, y, l, n=None): return self.get_omega(l)[self.get_block(x), self.get_block(y)]/self.get_n(n)

    def psi(self, x, y, l=None, n=None):
        if l is None: return np.exp(-self._omega_cumsum[:, self.get_block(x), self.get_block(y)]/self.get_n(n))
        else: return super().psi(x, y, l, n)

    def foo_critical(self, x, y, pi):
        #spectral condition for 2-block SBM
        if isiterable(x): return [self.foo_critical(x_, y, pi) for x_ in x]
        assert(0<=pi<=0.5)
        if x>2:
            if isiterable(y): return [np.nan]*len(y) #to not compute other branch
            else: return np.nan
        else: return pi*(1-pi)*(y**2-x**2) + (x-1)

    def foo_size_analytic(self, x, y, pi, boolean=False):
        if isiterable(x): return [self.foo_size_analytic(x_, y, pi) for x_ in x]
        if isiterable(y): return [self.foo_size_analytic(x, y_, pi) for y_ in y]
        if boolean: return type(self)(B=[[x, y], [y, x]], pi=[pi, 1-pi], set_rho=True, set_omega=False).is_percolating()
        else: return type(self)(B=[[x, y], [y, x]], pi=[pi, 1-pi], set_rho=True, set_omega=False).rho()

    def foo_size_empirical(self, x, y, pi, samples=1, n=1000):
        if isiterable(x): return [self.foo_size_empirical(x_, y, pi, samples=samples, n=n) for x_ in x]
        if isiterable(y): return [self.foo_size_empirical(x, y_, pi, samples=samples, n=n) for y_ in y]
        tmp = type(self)(B=[[x, y], [y, x]], pi=[pi, 1-pi], n=n, set_rho=False, set_omega=False) #for quick computation, no need to compute rho/omega
        return tmp.rho_empirical(tmp.sample(samples))

    def critical_xy(self, fine=False):
        if fine: return (np.linspace(0, 4, 80), np.linspace(0, 4, 40))
        else: return (np.linspace(0, 4, 40), np.linspace(0, 4, 40))

    def critical_z(self, param=None, fine=False, num=50):
        if fine: return np.linspace(0, 0.5, num+1)
        else:
            if param is None: param = 0.2
            assert(0<param<1)
            return param

    def plot_critical(self, pi=0.2, xlabel='$c_\\mathrm{in}$', ylabel='$c_\\mathrm{out}$', title='SBM ($k=2$)', **kwargs):
        #returns param points to plot spectral condition
        return super().plot_critical(*self.critical_xy(), ax_kwargs=dict(equal=True, xlabel=xlabel, ylabel=ylabel, xscale='int', yscale='int', title=title), pi=pi, **kwargs)

    def plot_critical_z(self, pi=0.2, xlabel='$c_\\mathrm{in}$', ylabel='$c_\\mathrm{out}$', title='SBM ($k=2$)', cmap='Spectral', ax=None, pad=0.025, **kwargs):
        x, y = self.critical_xy(fine=True)
        ax_ = super().plot_critical_z(x, y, self.critical_z(fine=True), ax=ax, ax_kwargs=dict(equal=True, xlabel=xlabel, ylabel=ylabel, xscale='int', yscale='int', title=title), c_kwargs=dict(label='$\\pi$', cmap=cmap), **kwargs)
        if pi is not None:
            if ax is None: ax = ax_
            super().plot_critical(x, y, pi=pi, ax=ax, color='black')
            #draw_text(ax, '$\\pi=%.1f$'%pi, 'bottom-right', pad=pad)
            get_legend('$\\pi=%.1f$'%pi, 'black', loc=1, ax=ax)
        return ax

    def to_rankone(self, n=None):
        fooint = (self._rank1_beta*(self._rank1_foo@self.pi), 
            np.log(self._rank1_beta)+np.log(self._rank1_foo)@self.pi, 
            self._rank1_beta**2*(self._rank1_foo**2@self.pi))
        return RankOneGraph(foo=lambda x: self._rank1_beta*self._rank1_foo[self.get_block(x)], fooint=fooint, name=self.name+'_rank1', n=self.get_n(n))

    def to_spd(self, n=None): return spd(self.B, self.get_n(n), self.pi)

    def plot_aspl(self, samples=dict(), d=4, pi=[0, 0.1, 0.2, 0.3, 0.4, 0.5], num=10, ax=None, figsize=(4, 3), dpi=180, cmap='coolwarm', xlabel='Homophily $\\propto(p_{in}-p_{out})$', ylabel='Mean Geodesic Length $\\langle\\lambda\\rangle$', title='', **kwargs):
        no_axis = ax is None
        ax = get_axes(ax=ax, figsize=figsize, dpi=dpi, xlabel=xlabel, ylabel=ylabel, title=title)
        if no_axis: colors = get_color(pi, ax=ax, label='$\\pi$', cmap=cmap)
        else: colors = get_color(pi, label='$\\pi$', cmap=cmap)
        for p in pi:
            if p not in samples: samples[p] = []
        def get_params(p):
            q = 2*p*(1-p)
            delta = np.hstack([np.linspace(-d/q, 0, num+1), np.linspace(0, d/(1-q), num+1)[1:]])
            x = d+delta*(q-1)
            return x, delta
        for idx in range(len(pi)):
            if pi[idx]==0: continue
            x, y = get_params(pi[idx])
            for k in range(2*num):
                i, j = x[k], y[k]
                tmp = type(self)(B=[[i+j, i], [i, i+j]], pi=[pi[idx], 1-pi[idx]], set_rho=False, set_omega=True)
                if j>0: j = j/y[-1]
                elif j<0: j = -j/y[0]
                samples[pi[idx]].append([j, tmp.aspl(), tmp.to_rankone().aspl()])
            samples[pi[idx]] = np.array(samples[pi[idx]])
            ax.plot(samples[pi[idx]][:,0], samples[pi[idx]][:,1], color=colors[idx], marker='$\\circ$')
            ax.plot(samples[pi[idx]][:,0], samples[pi[idx]][:,2], color=colors[idx], marker='$\\times$')        
        if no_axis:
            if samples: return samples, ax
            else: return ax
        elif samples: return samples

class GRGG1(RandomGraph):

    def __init__(self, d=4, beta=None, scale=0.1, num=2**8, name='grgg_1', **kwargs):
        from scipy.stats import norm
        assert(scale>0)
        self.scale = float(scale)
        self.set_beta(d, beta)
        RandomGraph.__init__(self, mu=norm(), nu=lambda x, y: self.beta*np.exp(-(x-y)**2/(2*self.scale)), num=num, name=name, **kwargs)

    def get_beta(self, d): return d*np.sqrt(2/self.scale+1)

    def set_beta(self, d=4, beta=None):
        if beta is None:
            if d is None: raise ValueError('expected value for either "beta" or "d" (mean degree)')
            assert(d>0)
            self.beta = self.get_beta(d)
        elif d is None:
            assert(beta>0)
            self.beta = float(beta)
        else: raise ValueError('expected value for either "beta" or "d" (mean degree)')

class GRGG(RandomGraph):
    
    def __init__(self, d=4, beta=None, scale=[[0.08, 0.04], [0.04, 0.08]], k=2, num=2**12, name='grgg', rho_interpolate=False, **kwargs):
        from scipy.stats import multivariate_normal, norm
        self.set_scale(scale, k)
        self.set_beta(d, beta)
        self._mu_marginal = norm()
        self.rho_interpolate = bool(rho_interpolate)
        if self.k==1:
            def kernel(x, y):
                d = x-y
                return self.beta*np.exp(-d**2*self.scale_inv[0,0]/2)
        else:
            def kernel(x, y):
                d = np.array(x)-np.array(y)
                return self.beta*np.exp(-d@self.scale_inv@d/2)
        RandomGraph.__init__(self, mu=multivariate_normal(mean=[0.]*k), nu=kernel, x=((-float('inf'), float('inf')),)*k, num=num, name=name, **kwargs)

    def get_beta(self, d): return d*np.sqrt((self._scale_inv_plus_one_det*self._scale_plus_one_inv_plus_one_det))

    def set_beta(self, d=4, beta=None):
        if beta is None:
            if d is None: raise ValueError('expected value for either "beta" or "d" (mean degree)')
            assert(d>0)
            self.beta = self.get_beta(d)
        elif d is None:
            assert(beta>0)
            self.beta = float(beta)
        else: raise ValueError('expected value for either "beta" or "d" (mean degree)')

    def set_scale(self, scale=1, k=2):
        def isposdef(x): return np.all(np.linalg.eigvals(x)>0)
        if not isiterable(scale):
            if not scale>0: raise ValueError('expected scale to be a positive value')
            assert(k>0)
            scale, scale_inv, scale_eigval, scale_eigvec = scale*np.eye(k), np.eye(k)/scale, scale*np.ones(k), np.eye(k)
        else:
            scale = np.array(scale)
            if isiterable_till(scale, 1):
                if not np.all(scale>0): raise ValueError('expected scale to be a positive vector')
                scale, scale_inv, scale_eigval, scale_eigvec = np.diag(scale), np.diag(1/scale), scale, np.eye(len(scale))
            elif isiterable_till(scale, 2):
                if not np.all(scale==scale.T): raise ValueError('expected scale to be a symmetric matrix')
                scale_eigval, scale_eigvec = np.linalg.eigh(scale)
                if not np.all(scale_eigval>0): raise ValueError('expected scale to be a positive definite matrix')
                scale_inv = np.linalg.inv(scale)
            else: raise ValueError('expected scale to be a positive value/vector/matrix indicating connection radii')
        self.scale = scale
        self.scale_inv = scale_inv
        idx = np.argsort(scale_eigval)[::-1]
        self.scale_eigval = scale_eigval[idx]
        self.scale_eigvec = scale_eigvec[:, idx]
        self.k = len(scale)
        self._scale_plus_one = np.eye(self.k) + self.scale
        self._scale_plus_one_inv = np.linalg.inv(self._scale_plus_one)
        self._scale_det = np.linalg.det(self.scale)
        self._scale_inv_plus_one_det = np.linalg.det(np.eye(self.k)+ self.scale_inv)
        self._scale_plus_one_inv_plus_one_det = np.linalg.det(np.eye(self.k)+ self._scale_plus_one_inv)

    def transform(self, x): return self._mu_marginal.cdf(x)

    def transform_inv(self, x): return self._mu_marginal.ppf(x)

    def isdiag(self):
        i, j = np.nonzero(self.scale)
        return np.all(i==j)

    def iseye(self):
        if not self.isdiag(): return False
        x = np.diag(self.scale)
        return x.min()==x.max()

    def set_omega(self, **kwargs): self.set_coeffs()

    def set_coeffs(self):
        c, U, V = [0., self.beta], [np.zeros((self.k, self.k)), np.eye(self.k)], [np.zeros((self.k, self.k)), np.eye(self.k)]
        for i in range(self.lmax-1):
            tmp = np.linalg.inv(self._scale_plus_one + U[-1])
            c.append(c[-1]*self.beta*np.sqrt(self._scale_det*np.linalg.det(tmp)))
            U.append(np.eye(self.k) - tmp)
            V.append(V[-1]@tmp)
        self._c, self._U, self._V = np.array(c), np.array([self.scale_inv@u for u in U]), np.array([self.scale_inv@v for v in V])

    def get_coeffs(self, l=None):
        if l is None: return self._c, self._U, self._V
        elif 0<=l<=self.lmax: return self._c[l], self._U[l], self._V[l]
        else: raise ValueError('expected "l" to be between 0 and lmax (%i)'%self.lmax)

    def degree(self, x=None):
        if x is None: return self.beta/np.sqrt((self._scale_inv_plus_one_det*self._scale_plus_one_inv_plus_one_det))
        else:
            x = np.array(x)
            if x.ndim==1: return self.beta*np.exp(-x**2*self._scale_plus_one_inv[0,0]/2)/np.sqrt(self._scale_inv_plus_one_det)
            else: return self.beta*np.exp([-i@self._scale_plus_one_inv@i/2 for i in x])/np.sqrt(self._scale_inv_plus_one_det)

    def set_rho(self, space=(-3,3)):
        if self.rho_interpolate: warn('setting interpolator to compute percolation probabilities may take some time; try changing "num"')
        else: warn('setting gaussian curve parameters to compute percolation probabilities may take some time; try changing "num"')
        xs = self.get_nodes(space=space)
        probs = self.mu(xs)
        probs = probs/probs.sum()
        kernel = np.array([[self._nu(x, y) for y in xs] for x in xs])
        def foo(rho=0.): return 1 - np.exp(-np.dot(kernel, rho*probs)) #consistency equation for size of GCC
        rho_old, rho_new = np.ones(len(xs)), np.ones(len(xs))
        while True:
            rho_new = foo(rho_old)
            if ((rho_new-rho_old)**2).mean()<self.tol: break
            rho_old = rho_new
        if self.rho_interpolate:
            if self.k==1: self._rho = self.get_interpolator(xs, rho_new)
            elif self.k==2: self._rho = self.get_interpolator(xs[:,0], xs[:,1], rho_new)
            else: raise NotImplementedError('computing percolation probabilities via interpolation not yet implemented for GRGGs with k>2')
        else:
            from scipy.optimize import curve_fit
            param_mu, param_sigma = curve_fit(self.foo_generalized_gaussian, xs, rho_new, p0=(1,)*(2+self.k), bounds=((0, 1)+(0,)*self.k, (1,)+(np.inf,)*(1+self.k)))
            self._rho = tuple(param_mu)
    
    def foo_generalized_gaussian(self, x, a, b, *args):
        if self.k==1: return a*np.exp(-(args[0]*np.abs(x)**2)**b)
        else:
            x = np.array(x)
            return a*np.exp(-np.diag((x@(self.scale_eigvec@np.diag(args)@self.scale_eigvec.T)@x.T))**b)

    def rho(self, x=None):
        if self.rho_interpolate: return super().rho(x)
        if x is None: return self.foo_generalized_gaussian(self._mu.rvs(size=self.num), *self._rho).mean()
        else: return self.foo_generalized_gaussian(x, *self._rho)

    def is_percolating(self):
        if self.rho_interpolate: super().is_percolating()
        else: return not(np.isclose(self._rho[0], 0))

    def omega(self, x, y, l, n=None):
        x, y = np.array(x), np.array(y)
        c, U, V = self.get_coeffs(l)
        return c*np.exp(-(x@U@x+y@U@y)/2+x@V@y)/self.get_n(n)

    def psi(self, x, y, l=None, n=None):
        if l is None:
            if self.k==1: x, y = np.array([x]), np.array([y])
            else: x, y = np.array(x), np.array(y)
            c, U, V = self.get_coeffs()
            return np.exp(-(c*np.exp(-(self.dot_long(x, x, U) + self.dot_long(y, y, U))/2 + self.dot_long(x, y, V))).cumsum()/self.get_n(n))
        else: return super().psi(x, y, l, n)

    def foo_critical(self, x, y, beta):
        #spectral condition for 2D GRGG
        if isiterable(x): return [self.foo_critical(x_, y, beta) for x_ in x]
        assert(beta>0)
        def foo(x): return x+(1+np.sqrt(1+4*x))/2
        return foo(x)*foo(y) - beta**2

    def foo_size_analytic(self, x, y, beta, boolean=False):
        if isiterable(x): return [self.foo_size_analytic(x_, y, beta) for x_ in x]
        if isiterable(y): return [self.foo_size_analytic(x, y_, beta) for y_ in y]
        try:
            if boolean: return type(self)(d=None, beta=beta, scale=[1/x, 1/y], k=2, num=2**8, set_rho=True, set_omega=False).is_percolating()
            else: return type(self)(d=None, beta=beta, scale=[1/x, 1/y], k=2, num=2**8, set_rho=True, set_omega=False).rho()
        except: return 0

    def foo_size_empirical(self, x, y, beta, samples=1, n=1000):
        if isiterable(x): return [self.foo_size_empirical(x_, y, beta, samples=samples, n=n) for x_ in x]
        if isiterable(y): return [self.foo_size_empirical(x, y_, beta, samples=samples, n=n) for y_ in y]
        tmp = type(self)(d=None, beta=beta, scale=[1/x, 1/y], k=2, n=n, num=2**8, set_rho=False, set_omega=False) #for quick computation, no need to compute rho/omega
        return tmp.rho_empirical(tmp.sample(samples))

    def critical_xy(self, fine=False):
        if fine: return (np.linspace(0, 8, 80), np.linspace(0, 8, 80))
        else: return (np.linspace(0, 8, 40), np.linspace(0, 8, 40))

    def critical_z(self, param=None, fine=False, num=56):
        if fine: return np.linspace(1, 12, num)
        else:
            if param is None: param = 4
            assert(param>0)
            return param

    def plot_critical(self, beta=4, xlabel='$r_1$', ylabel='$r_2$', title='', **kwargs):
        #returns param points to plot spectral condition
        return super().plot_critical(*self.critical_xy(), ax_kwargs=dict(equal=True, xlabel=xlabel, ylabel=ylabel, xscale='int', yscale='int', title=title), beta=beta, **kwargs)

    def plot_critical_z(self, beta=4, xlabel='$r_1$', ylabel='$r_2$', title='GRGG ($k=2$)', cmap='Spectral', ax=None, pad=0.025, **kwargs):
        x, y = self.critical_xy(fine=True)
        ax_ = super().plot_critical_z(x, y, self.critical_z(fine=True), ax=ax, ax_kwargs=dict(equal=True, xlabel=xlabel, ylabel=ylabel, xscale='int', yscale='int', title=title), c_kwargs=dict(label='$\\beta$', cmap=cmap), **kwargs)
        if beta is not None:
            if ax is None: ax = ax_
            super().plot_critical(x, y, beta=beta, ax=ax, color='black')
            #draw_text(ax, '$\\beta=%i$'%beta, 'bottom-right', pad=pad)
            get_legend('$\\beta=%i$'%beta, 'black', loc=1, ax=ax)
        return ax

class RankOneGraph(RandomGraph):

    def __init__(self, mu=None, foo=None, logfoo=None, fooint=None, name='rank_one_graph', **kwargs):
        self.set_foo(foo, logfoo, fooint, mu)
        RandomGraph.__init__(self, mu=mu, nu=lambda x, y: self.foo(x)*self.foo(y), name=name, **kwargs)
        self._foo2int_minus_one = self._foo2int-1
        self._euler = 0.5772156649015329
        self._log_foo2int = np.log(self._foo2int)
        self._log_foo2int_minus_euler = np.log(self._foo2int_minus_one) - self._euler

    def set_foo(self, foo=None, logfoo=None, fooint=None, mu=None):
        if foo is None: foo = 2.
        if isinstance(foo, (float, int)): self.foo = lambda x: float(foo)
        else:
            assert(callable(foo))
            self.foo = foo
        if logfoo is None: self.logfoo = lambda x: np.log(self.foo(x))
        else:
            assert(callable(logfoo))
            self.logfoo = logfoo
        if fooint is None: self.set_fooint(mu=mu)
        else:
            assert(isiterable_till(fooint, 1) and isnumeric(fooint) and len(fooint)==3)
            self._fooint, self._logfooint, self._foo2int = tuple(fooint)

    def set_fooint(self, mu=None):
        if mu is None:
            from scipy.stats import uniform
            mu = uniform()
        self._fooint = mu.expect(self.foo)
        self._logfooint = mu.expect(self.logfoo)
        self._foo2int = mu.expect(lambda x: self.foo(x)**2)

    def degree(self, x=None):
        if x is None: return self._fooint**2
        else:
            x = np.array(x)
            return self._fooint*self.foo(x)

    def set_rho(self):
        def foo(rho=0.): return self._fooint - self.expect(lambda x: np.exp(-self.foo(x)*rho)*self.foo(x)) #consistency equation for size of GCC
        rho_old, rho_new = 1, 1
        while True:
            rho_new = foo(rho_old)
            if (rho_new-rho_old)**2<self.tol: break
            rho_old = rho_new
        self._rho = float(rho_new)

    def rho(self, x=None):
        def foo(rho=0.): return 1 - np.exp(-rho*self.foo(x)*self._rho) #consistency equation for size of GCC
        if x is None: return 1 - self.expect(lambda x: np.exp(-self.foo(x)*self._rho))
        else:
            if not isiterable(x): x = [x]
            x = np.array(x)
            num = len(x)
            rho_old, rho_new = np.ones(num), np.ones(num)
            while True:
                rho_new = foo(rho_old)
                if ((rho_new-rho_old)**2).mean()<self.tol: break
                rho_old = rho_new
            if num==1: return rho_new[0]
            else: return rho_new

    def is_percolating(self): return not(np.isclose(self._rho, 0))

    def set_omega(self, **kwargs):
        #for faster computation of psi and average spl
        out = [0, 1]
        for i in range(self.lmax-1): out.append(out[-1]*self._foo2int)
        self._omega = np.array(out)
        self._omega_cumsum = self._omega.cumsum(axis=0)

    def get_omega(self, l=None):
        if l is None: return self._omega
        elif 0<=l<=self.lmax: return self._omega[l]
        else: raise ValueError('expected "l" to be between 0 and lmax (%i)'%self.lmax)

    def omega(self, x, y, l, n=None): return self.foo(x)*self.foo(y)*self.get_omega(l)/self.get_n(n)

    def psi(self, x, y, l=None, n=None):
        if l is None: return np.exp(-self.foo(x)*self.foo(y)*self._omega_cumsum/self.get_n(n))
        else: return super().psi(x, y, l, n)

    def spl_analytic(self, x, y, n=None): return 0.5+(self._log_foo2int_minus_euler-self.lognu(x, y, n))/self._log_foo2int

    def aspl_analytic(self, x=None, n=None, bound=False):
        if x is None:
            if bound: return 0.5+(self._log_foo2int_minus_euler-2*np.log(self._fooint)+np.log(self.get_n(n)))/self._log_foo2int
            else: return 0.5+(self._log_foo2int_minus_euler-2*self._logfooint+np.log(self.get_n(n)))/self._log_foo2int
        else:
            if bound: return 0.5+(self._log_foo2int_minus_euler-self.logfoo(x)-np.log(self._fooint)+np.log(self.get_n(n)))/self._log_foo2int
            else: return 0.5+(self._log_foo2int_minus_euler-self.logfoo(x)-self._logfooint+np.log(self.get_n(n)))/self._log_foo2int

    def closeness_analytic(self, x, n=None): return 1/self.aspl_analytic(x=x, n=n)

    def eigenmean_analytic(self, n=None): return(self._log_foo2int, self._logfooint-0.5*np.log(self.get_n(n)))

class ScaleFree(RankOneGraph):

    def __init__(self, d=4, beta=None, h=None, alpha=0.5, n=2**9, name='scale_free', **kwargs):
        from scipy.stats import uniform
        assert(0<alpha<=1)
        self.alpha = float(alpha)
        self.set_beta(d, beta, h, n)
        RankOneGraph.__init__(self, mu=uniform(loc=self.h, scale=1-self.h), foo=lambda x: self.sqrtbeta*(self.h/x)**alpha, x=((self.h, 1.),), n=n, name=name, **kwargs)

    def set_fooint(self, **kwargs):
        if self.alpha==1: self._fooint = -self.sqrtbeta*self.h*self.logh/(1-self.h)
        else: self._fooint = self.sqrtbeta*(self.h**self.alpha-self.h)/((1-self.alpha)*(1-self.h))
        self._logfooint = 0.5*self.logbeta + self.alpha*(1+self.logh/(1-self.h))
        if self.alpha==0.5: self._foo2int = -self.beta*self.h*self.logh/(1-self.h)
        else: self._foo2int = self.beta*(self.h**(2*self.alpha)-self.h)/((1-2*self.alpha)*(1-self.h))

    def get_beta(self, d):
        if self.alpha==1: return d*((1-self.h)/(self.h*self.logh))**2
        else: return d*((1-self.alpha)*(1-self.h)/(self.h**self.alpha-self.h))**2

    def get_h(self, d):
        if self.alpha==1:
            def foo(logh): return d-self.beta*(np.exp(logh)*logh/(1-np.exp(logh)))**2
        else:
            def foo(logh): return d-self.beta*((np.exp(logh*self.alpha)-np.exp(logh))/((1-self.alpha)*(1-np.exp(logh))))**2
        from scipy.optimize import brentq
        try: param, obj = brentq(foo, a=-500, b=-1, full_output=True)
        except ValueError: raise ValueError('unable to find optimal value of "h" for given d, beta and alpha; consider a different alpha or n')
        if not obj.converged: raise ValueError('unable to find optimal value of "h" for given d, beta and alpha; consider a different alpha or n')
        return np.exp(param)

    def critical_h(self):
        if self.alpha==0.5:
            def foo(logh): return 1+self.beta*np.exp(logh)*logh/(1-np.exp(logh))
        else:
            def foo(logh): return 1-self.beta*(np.exp(2*logh*self.alpha)-np.exp(logh))/((1-2*self.alpha)*(1-np.exp(logh)))
        from scipy.optimize import brentq
        try: param, obj = brentq(foo, a=-500, b=-1, full_output=True)
        except ValueError: raise ValueError('unable to find critical value of "h"')
        if not obj.converged: raise ValueError('unable to find critical value of "h"')
        return np.exp(param)

    def critical_d(self): return type(self)(h=self.critical_h(), alpha=self.alpha, beta=self.beta, d=None, n=self.n, set_rho=False, set_omega=False).degree()

    def set_beta(self, d=4, beta=None, h=None, n=2**9):
        if beta is None:
            if d is None: raise ValueError('expected value for either "beta" or "d" (mean degree)')
            assert(d>0)
            if h is None:
                assert(n>0)
                self.beta = float(n) # "un"-sparsify
                self.h = self.get_h(d)
            else:
                assert(0<h<1)
                self.h = float(h)
                self.beta = self.get_beta(d)
        elif d is None:
            assert(beta>0)
            self.beta = float(beta)
            assert(0<h<1)
            self.h = float(h)
        else: raise ValueError('expected value for either "beta" or "d" (mean degree)')
        self.sqrtbeta = np.sqrt(self.beta)
        self.logbeta = np.log(self.beta)
        self.logh = np.log(self.h)

    def degree_analytic(self, k=None, kmin=1, kmax=float('inf'), exact=False):
        lower = self._fooint*self.sqrtbeta*self.h**self.alpha
        upper = self._fooint*self.sqrtbeta
        tau_ = 1/self.alpha
        tau = 1+tau_
        factor = tau_*lower**tau_/(1-self.h)
        def apx_deg(k): return factor*k**(-tau)
        def deg(k):
            from scipy.special import gamma, gammaincc
            try: return factor*gamma(k-tau_)*(gammaincc(k-tau_, lower)-gammaincc(k-tau_, upper))/gamma(k+1)
            except: return np.nan
        no_k = k is None
        if no_k: k = np.arange(max(kmin, 1+min(np.floor(tau_), np.floor(lower))), min(kmax, np.floor(upper)))
        if exact: dist = [deg(i) for i in k]
        else: dist = [apx_deg(i) for i in k]
        dist = np.array(dist)
        dist[dist==0] = np.nan
        if no_k: return (k, dist)
        else: return dist

    def plot_spl_alpha(self, samples=False, d=None, logn=range(7, 13), ax=None, figsize=(4, 3), dpi=180, cmap='coolwarm', marker='$\\circ$', size=5, xlabel='$\\alpha$', ylabel='Mean Geodesic Length $\\langle\\lambda\\rangle$', title='', **kwargs):
        no_axis = ax is None
        ax = get_axes(ax=ax, figsize=figsize, dpi=dpi, xlabel=xlabel, ylabel=ylabel, title=title)
        if d is None: d = self.degree()
        if isinstance(samples, bool) and samples: samples = {2**i:dict() for i in logn}
        ns = [2**i for i in logn]
        alphas = [0.01] + [0.1*i for i in range(1, 11)]
        if no_axis: colors = get_color(ns, ax=ax, scale='log2', label='$n$', cmap=cmap)
        else: colors = get_color(ns, scale='log2', label='$n$', cmap=cmap)
        for i in range(len(ns)):
            y, y_err, y_hat = [], [], []
            if samples and ns[i] not in samples: samples[ns[i]] = {}
            for a in alphas:
                tmp = type(self)(d=d, alpha=a, n=ns[i], set_rho=False, set_omega=False, **kwargs)
                y_hat.append(tmp.aspl_analytic())
                if samples:
                    if a not in samples[ns[i]]: samples[ns[i]][a] = np.array(tmp.spl_empirical(tmp.sample(gcc=True), mean_network=True))
                    x_mu, x_sigma = samples[ns[i]][a].mean(), samples[ns[i]][a].std()
                    y.append(x_mu)
                    y_err.append(x_sigma)
            if samples: ax.errorbar(alphas, y, yerr=y_err, color=colors[i], ls='', marker=marker, ms=size)
            ax.plot(alphas, y_hat, color=colors[i])
        if no_axis:
            if samples: return samples, ax
            else: return ax
        elif samples: return samples

    def plot_critical_alpha(self, logn=range(7, 13), ax=None, figsize=(4, 3), dpi=180, cmap='Blues', xlabel='$\\alpha$', ylabel='$n$', zlabel='$d$', title='', **kwargs):
        no_axis = ax is None
        ax = get_axes(ax=ax, figsize=figsize, dpi=dpi, xlabel=xlabel, ylabel=ylabel, yscale='log2', title=title)
        ns = [2**i for i in logn]
        alphas = [0.01] + [0.1*i for i in range(1, 11)]
        crit_d = np.zeros((len(ns), len(alphas)))
        for i in range(len(ns)):
            for j in range(len(alphas)):
                try: crit_d[i,j] = type(self)(d=None, h=0.1, beta=ns[i], alpha=alphas[j], n=ns[i], set_rho=False, set_omega=False, **kwargs).critical_d()
                except: crit_d[i,j] = np.nan
        cb = ax.contourf(alphas, ns, crit_d, cmap=cmap)
        ax.get_figure().colorbar(cb, ax=ax, label=zlabel)
        if no_axis: return ax, crit_d
        else: return crit_d

    def foo_critical(self, x, y, alpha=0.5):
        #spectral condition for a scale-free graphon
        if isiterable(x): return [self.foo_critical(x_, y, alpha) for x_ in x]
        assert(0<alpha<=1)
        if alpha==0.5: return x*np.log(np.sqrt(4*y/x)-1)/(2*(1-np.sqrt(x/(4*y)))**2) - 1
        else:
            if isiterable(y): return [self.foo_critical(x, y_, alpha) for y_ in y]
            try: return type(self)(d=x, beta=None, h=None, alpha=alpha, n=y, set_rho=False, set_omega=False)._foo2int - 1
            except: return np.nan

    def foo_size_analytic(self, x, y, alpha, boolean=False):
        if isiterable(x): return [self.foo_size_analytic(x_, y, alpha) for x_ in x]
        if isiterable(y): return [self.foo_size_analytic(x, y_, alpha) for y_ in y]
        try:
            if boolean: return type(self)(d=x, beta=None, h=None, alpha=alpha, n=y, set_rho=True, set_omega=False).is_percolating()
            else: return type(self)(d=x, beta=None, h=None, alpha=alpha, n=y, set_rho=True, set_omega=False).rho()
        except: return 0

    def foo_size_empirical(self, x, y, alpha, samples=1, n=1000):
        if isiterable(x): return [self.foo_size_empirical(x_, y, alpha, samples=samples, n=n) for x_ in x]
        if isiterable(y): return [self.foo_size_empirical(x, y_, alpha, samples=samples, n=n) for y_ in y]
        try:
            tmp = type(self)(d=x, beta=None, h=None, alpha=alpha, n=y, set_rho=False, set_omega=False) #for quick computation, no need to compute rho/omega
            return tmp.rho_empirical(tmp.sample(samples))
        except:
            if samples>1: return [np.nan]*samples
            else: return np.nan

    def critical_xy(self, fine=False):
        if fine: return (np.linspace(0, 1.25, 50), 2**np.arange(7, 13))
        else: return (np.linspace(0, 4, 40), 2**np.arange(7, 13))

    def critical_z(self, param=None, fine=False, num=50):
        if fine: return np.linspace(0, 1, num+1)[1:]
        else:
            if param is None: param = 0.5
            assert(0<param<=1)
            return param

    def plot_critical(self, alpha=0.5, xlabel='$\\langle d\\rangle$', ylabel='$n$', title='Scale-free Graphon', ax=None, **kwargs):
        #returns param points to plot spectral condition
        x, y = self.critical_xy()
        ax_ = super().plot_critical(x, y, ax=ax, ax_kwargs=dict(equal=False, xlabel=xlabel, ylabel=ylabel, xscale='int', yscale='log2', title=title), alpha=alpha, **kwargs)
        if alpha==0.5 or alpha==1:
            if ax is None: ax = ax_
            y_ = np.linspace(y.min(), y.max(), 1000)
            if alpha==0.5: x_ = 4/(np.log(y_)+np.log(np.log(y_)))
            else: x_ = (np.log(y_)-np.log(np.log(y_)))**2/y_#4*(np.log(y_/2)+np.log(np.log(y_/2)))/y_
            ax.plot(x_, y_, linestyle='dotted', color='red')
        return ax

    def plot_critical_z(self, alpha=0.5, xlabel='$\\langle d\\rangle$', ylabel='$n$', title='Scale-free Graphon', cmap='Spectral', ax=None, pad=0.025, **kwargs):
        x, y = self.critical_xy(fine=True)
        ax_ = super().plot_critical_z(x, y, self.critical_z(fine=True), ax=ax, ax_kwargs=dict(equal=False, xlabel=xlabel, ylabel=ylabel, xscale='int', yscale='log2', title=title), c_kwargs=dict(label='$\\alpha$', cmap=cmap), **kwargs)
        if alpha is not None:
            if ax is None: ax = ax_
            super().plot_critical(x, y, alpha=alpha, ax=ax, color='black')
            #draw_text(ax, '$\\alpha=%.1f$'%alpha, 'bottom-right', pad=pad)
            get_legend('$\\alpha=%.1f$'%alpha, 'black', loc=1, ax=ax)
        return ax

class MaxGraphon(RandomGraph):

    def __init__(self, beta=8, name='max_graphon', num=2**8, **kwargs):
        from scipy.stats import uniform
        assert(beta>0)
        self.beta = float(beta)
        RandomGraph.__init__(self, mu=uniform(), nu=lambda x, y: self.beta*(1-max(x, y)), num=num, name=name, **kwargs)

class SimplexRDPG(RandomGraph):

    def __init__(self, d=4, alpha=[0.8, 0.8, 2], num=2**12, name='simplex_rdpg', **kwargs):
        from scipy.stats import dirichlet
        self.d = float(d)
        phi = np.array(alpha) #default for k=1 is alpha=[0.8, 1.2]
        self.phi = phi/phi.sum()
        self.beta = self.d/(self.phi**2).sum()
        self.sqrt3by2 = np.sqrt(3)/2
        RandomGraph.__init__(self, mu=dirichlet(alpha=alpha), nu=lambda x, y: self.beta*np.dot(x, y), k=len(alpha), k_eff=len(alpha)-1, num=num, name=name, **kwargs)

    def nullmodal(self): return (self._mu.alpha<1).any()

    def get_nodes(self, transform=False):
        flag = self.nullmodal()
        num = self.num
        if self.k==1: points = np.zeros((num, 1))
        elif self.k==2:
            points = np.linspace(0, 1, num)[:, None]
            if flag: points = points[1:-1,:]
        elif self.k==3: 
            x = np.linspace(0, 1, num)
            dy = self.sqrt3by2/(num-1)
            points = []
            if flag: x = (x[1:-2]+x[2:-1])/2
            for i in range(int(flag), num-1):
                y = i*dy*np.ones(len(x))
                points += list(zip(x, y))
                x = (x[:-1]+x[1:])/2
            if not flag: points.append((1/2, self.sqrt3by2))
            points = np.array(points)
        else: raise NotImplementedError('node location access not implemented for k=%i'%self.k)
        if not transform: points = self.transform_inv(points)
        return points

    def transform(self, x):
        if isiterable_till(x, 1):
            assert(len(x)==self.k)
            x = np.array(x)
            x = x/x.sum()
            assert((x>=0).all())
            if self.k==1: return np.array([0.])
            elif self.k==2: return np.array([x[1]])
            elif self.k==3: return np.array([x[1]+x[2]/2, self.sqrt3by2*x[2]])
            else: raise NotImplementedError('transforming coordinates not yet implemented for k=%i'%self.k)
        else: return np.array([self.transform(i) for i in x])

    def transform_inv(self, x):
        if isiterable_till(x, 1):
            assert(len(x)==max(self.k-1, 1))
            x = np.array(x)
            if self.k==1: out = np.array([1-x[0]])
            elif self.k==2: out = np.array([1-x[0], x[0]])
            elif self.k==3:
                x_3 = x[1]/self.sqrt3by2
                x_2 = x[0]-x_3/2
                x_1 = 1-x_3-x_2
                out = np.array([x_1, x_2, x_3])
            else: raise NotImplementedError('inverse transforming coordinates not yet implemented for k=%i'%self.k)
            out[np.isclose(out, 0)] = 0. #floating-point error
            assert((out>=0).all())
            assert(np.allclose(out.sum(), 1))
            return out
        else: return np.array([self.transform_inv(i) for i in x])

    def plot_nodespace(self, space=None, transform=True, foo=None, foo_kwargs=dict(), **kwargs):
        x_t = self.get_nodes(transform=True)
        if foo is None: foo = self.mu
        x = self.transform_inv(x_t)
        y = np.array([foo(xx, **foo_kwargs) for xx in x])
        if self.k==2: return self.plot_curve(x_t[:,0], y, **kwargs)
        elif self.k==3: return self.plot_contour(x_t[:,0], x_t[:,1], y, tri=True, **kwargs)
        else: raise NotImplementedError('cannot plot in nodespace for k=%i'%self.k)

    def plot_edgespace(self, space=None, transform=True, foo=None, foo_kwargs=dict(), **kwargs):
        x_t = self.get_nodes(transform=True)
        if foo is None:
            foo = self.nu
            if 'n' not in foo_kwargs:
                foo_kwargs = foo_kwargs.copy()
                foo_kwargs['n'] = 1
        x = self.transform_inv(x_t)
        if self.k==2:
            y = np.array([[foo(xx, yy, **foo_kwargs) for yy in x] for xx in x]).T
            return self.plot_contour(x_t[:,0], x_t[:,0], y, **kwargs)
        elif self.k==3:
            yy = self.mean_location()
            y = np.array([foo(xx, yy, **foo_kwargs) for xx in x])
            return self.plot_contour(x_t[:,0], x_t[:,1], y, tri=True, **kwargs)
        else: raise NotImplementedError('cannot plot in edgespace for k=%i'%self.k)

    def degree(self, x=None):
        if x is None: return self.d
        else:
            x = np.array(x)
            if x.ndim==1: return self.beta*x@self.phi
            else: return self.beta*np.array([i@self.phi for i in x])

    def set_rho(self):
        def foo(rho=0.): return self.phi - (x*pi[:,None]*np.exp(-self.beta*x@rho[:,None])).sum(axis=0) #consistency equation for size of GCC
        x = self.get_nodes()
        pi = np.array([self.mu(i) for i in x])
        pi = pi/pi.sum()
        rho_old, rho_new = np.ones(self.k), np.ones(self.k)
        while True:
            rho_new = foo(rho_old)
            if ((rho_new-rho_old)**2).mean()<self.tol: break
            rho_old = rho_new
        self._rho = rho_new

    def rho(self, x=None):
        def foo(rho=0.): return 1 - np.exp(-self.beta*x@self._rho) #consistency equation for size of GCC
        if x is None:
            x, flag = self.get_nodes(), True
            pi = np.array([self.mu(i) for i in x])
            pi = pi/pi.sum()
        else: flag = False
        if not isiterable_till(x, 2): x = [x]
        x = np.array(x)
        num = len(x)
        rho_old, rho_new = np.ones(num), np.ones(num)
        while True:
            rho_new = foo(rho_old)
            if ((rho_new-rho_old)**2).mean()<self.tol: break
            rho_old = rho_new
        if flag: return rho_new@pi
        elif num==1: return rho_new[0]
        else: return rho_new

    def is_percolating(self): return not(np.allclose(self._rho, 0))

    def set_omega(self, **kwargs):
        #for faster computation of psi and average spl
        a = self._mu.alpha.sum()
        Phi = (a*self.beta/(a+1))*(self.phi[:,np.newaxis]*self.phi[np.newaxis,:] + np.diag(self.phi)/a)
        out = [np.zeros((self.k, self.k)), self.beta*np.eye(self.k)]
        for i in range(self.lmax-1): out.append(out[-1]@Phi)
        self._omega = np.array(out)
        self._omega_cumsum = self._omega.cumsum(axis=0)

    def get_omega(self, l=None):
        if l is None: return self._omega
        elif 0<=l<=self.lmax: return self._omega[l]
        else: raise ValueError('expected "l" to be between 0 and lmax (%i)'%self.lmax)

    def omega(self, x, y, l, n=None): return np.array(x)@self.get_omega(l)@np.array(y)/self.get_n(n)

    def psi(self, x, y, l=None, n=None):
        if l is None: return np.exp(-self.dot_long(x, y, self._omega_cumsum)/self.get_n(n))
        else: return super().psi(x, y, l, n)

class ExponentialRDPG(RandomGraph):
    #uses random fourier features (Rahimi and Recht 2007) to approximate a node2vec (Grover and Leskovec 2016) style embedding into RDPG framework
    def __init__(self, beta=2, a=0.8, b=0.8, k_features=2**8, num=2**5, name='exponential_rdpg', **kwargs):
        from scipy.stats import beta as beta_dist
        assert(beta>0)
        self.beta = float(beta)
        RandomGraph.__init__(self, mu=beta_dist(loc=-1, scale=2, a=a, b=b), nu=lambda x, y: self.beta*np.exp(np.dot(x, y)), x=((-1,1),), num=num, name=name, **kwargs)
        self.set_features(k_features)
        self.set_omega_features()

    def set_features(self, k_features=2**8):
        from scipy.stats import multivariate_normal, uniform
        k_features = int(k_features)
        assert(k_features>1)
        self._k = k_features
        self._w = multivariate_normal.rvs(mean=[0]*self.k, size=self._k)
        self._b = uniform.rvs(scale=2*np.pi, size=self._k)
        self._sqrt2byk = np.sqrt(2/self._k)
        x = self.get_nodes()
        pi = self.mu(x)
        pi[np.isinf(pi)] = 0
        pi = pi/pi.sum()
        x = self.get_features(x)
        self.phi = pi@x
        self.Phi = x.T@np.diag(pi)@x

    def get_features(self, x):
        if isiterable(x):
            x = np.array(x)
            if isiterable_till(x, 1):
                if self.k==1: return self._sqrt2byk*np.cos(x[:,None]*self._w[None,:]+self._b[None,:])*np.exp(x**2/2)[:,None]
                else: return self._sqrt2byk*np.cos(self._w@x+self._b)*np.exp(x@x/2)
            else: return self._sqrt2byk*np.cos(x@self._w.T+self._b[None,:])*np.exp([xx@xx/2 for xx in x])[:,None]
        elif self.k==1: return self._sqrt2byk*np.cos(x*self._w+self._b)*np.exp(x**2/2)
        else: raise ValueError('expected x to be of length %i'%self.k)

    def nu_features(self, x, y): return self.beta*max(0, np.dot(self.get_features(x), self.get_features(y)))# to correct for negatives; just in case

    def degree_features(self, x=None):
        if x is None: return self.beta*(self.phi**2).sum()
        else:
            x = self.get_features(x)
            if x.ndim==1: return self.beta*x@self.phi
            else: return self.beta*np.array([i@self.phi for i in x])

    def set_omega_features(self):
        #for faster computation of psi and average spl
        out = [np.zeros((self._k, self._k)), self.beta*np.eye(self._k)]
        for i in range(self.lmax-1): out.append(self.beta*out[-1]@self.Phi)
        self._omega_features = np.array(out)
        self._omega_features_cumsum = self._omega_features.cumsum(axis=0)

    def get_omega_features(self, l=None):
        if l is None: return self._omega_features
        elif 0<=l<=self.lmax: return self._omega_features[l]
        else: raise ValueError('expected "l" to be between 0 and lmax (%i)'%self.lmax)

    def omega_features(self, x, y, l, n=None): return self.get_features(x)@self.get_omega_features(l)@self.get_features(y)/self.get_n(n)

    def psi_features(self, x, y, l=None, n=None):
        if l is None: return np.exp(-self.dot_long(self.get_features(x), self.get_features(y), self._omega_features_cumsum)/self.get_n(n))
        else: return super().psi(x, y, l, n)

    def spl_features(self, x, y, n=None): return self.psi_features(x, y, n=n).sum()

    def plot_features(self, k_features=[2**2, 2**4, 2**6, 2**8, 2**10], dpi=180, figsize=2, cbar_pad=1.2):
        if not isiterable(k_features): k_features = [k_features]
        import matplotlib.pyplot as plt
        naxes = len(k_features)+1
        ncols = min(naxes, 3)
        nrows = naxes//3 + int(naxes%3!=0)
        fig, ax = plt.subplots(nrows=nrows*2, ncols=ncols, squeeze=False, dpi=dpi, figsize=(figsize*ncols*cbar_pad, figsize*nrows*2), constrained_layout=True)
        idx = 0
        for i in range(nrows):
            ax[i,0].set_ylabel('Kernel')
            ax[i+nrows,0].set_ylabel('Geodesic Length')
            for j in range(ncols):
                if idx<naxes-1:
                    self.set_features(k_features[idx])
                    self.set_omega_features()
                    self.plot_edgespace(foo=self.nu_features, ax=ax[i,j])
                    self.plot_edgespace(foo=self.spl_features, ax=ax[i+nrows,j], cmap='Purples')
                    for k in [0, nrows]: ax[i+k,j].set_title('$k=%i$'%k_features[idx])
                elif idx==naxes-1:
                    self.plot_edgespace(ax=ax[i,j])
                    self.plot_edgespace(foo=self.spl, ax=ax[i+nrows,j], cmap='Purples')
                    for k in [0, nrows]: ax[i+k,j].set_title('$\\nu(x,y)\\propto\\exp(xy)$')
                else: ax[i,j].set_axis_off()
                idx += 1
        return ax

class RandomRegularGraph(RankOneGraph):

    def __init__(self, d=3, name='random_regular', **kwargs):
        if not isinstance(d, int) or d<=0: raise ValueError('expected "d" to be a positive integer indicating degree of the d-regular graph but got "%s" instead'%str(d))
        self.d = d
        self.sqrtd = np.sqrt(self.d)
        self.logd = np.log(self.d)
        RankOneGraph.__init__(self, foo=lambda x: self.sqrtd, name=name, **kwargs)

    def set_fooint(self, **kwargs):
        self._fooint = self.sqrtd
        self._logfooint = 0.5*self.logd
        self._foo2int = self.d-1

    def sample(self, samples=1, n=None, gcc=False):
        if samples>1: return [self.sample(n=n, gcc=gcc) for i in range(int(samples))]
        import networkx as nx
        n = self.get_n(n)
        x = self._mu.rvs(size=n)
        G = nx.generators.random_graphs.random_regular_graph(d=self.d, n=n)
        for i in range(n):
            G.nodes[i]['x'] = x[i]
        if gcc:
            comp = sorted(nx.connected_components(G), key=len, reverse=True)
            for i in range(1, len(comp)): G.remove_nodes_from(comp[i])
        return G

    def plot_spl_d(self, samples=False, logn=range(7, 13), ax=None, figsize=(4, 3), dpi=180, cmap='coolwarm', marker='$\\circ$', size=5, xlabel='$d$', ylabel='Mean Geodesic Length $\\langle\\lambda\\rangle$', title='', **kwargs):
        no_axis = ax is None
        ax = get_axes(ax=ax, figsize=figsize, dpi=dpi, xlabel=xlabel, ylabel=ylabel, title=title)
        if isinstance(samples, bool) and samples: samples = {2**i:dict() for i in logn}
        ns = [2**i for i in logn]
        degrees = list(range(3, 10))
        if no_axis: colors = get_color(ns, ax=ax, scale='log2', label='$n$', cmap=cmap)
        else: colors = get_color(ns, scale='log2', label='$n$', cmap=cmap)
        for i in range(len(ns)):
            y, y_err, y_hat = [], [], []
            if samples and ns[i] not in samples: samples[ns[i]] = {}
            for d in degrees:
                tmp = type(self)(d=d, n=ns[i], set_rho=False, set_omega=False, **kwargs)
                y_hat.append(tmp.aspl_analytic())
                if samples:
                    if d not in samples[ns[i]]: samples[ns[i]][d] = np.array(tmp.spl_empirical(tmp.sample(gcc=True), mean_network=True))
                    x_mu, x_sigma = samples[ns[i]][d].mean(), samples[ns[i]][d].std()
                    y.append(x_mu)
                    y_err.append(x_sigma)
            if samples: ax.errorbar(degrees, y, yerr=y_err, color=colors[i], ls='', marker=marker, ms=size)
            ax.plot(degrees, y_hat, color=colors[i])
        if no_axis:
            if samples: return samples, ax
            else: return ax
        elif samples: return samples
