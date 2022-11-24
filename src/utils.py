from warnings import warn
from collections.abc import MutableSequence
import numpy as np

def isiterable(obj):
    if isinstance(obj, str): return False
    try:
        iter(obj)
        return True
    except TypeError: return False

def isiterable_till(obj, depth=1):
    assert(depth>=0)
    if depth==0: return not isiterable(obj)
    else: return isiterable(obj) and all([isiterable_till(o, depth=depth-1) for o in obj])

def istype(obj, type):
    if isiterable(obj): return all([istype(o, type) for o in obj])
    if isinstance(obj, type): return True
    else: return False

def isnumeric(obj): return istype(obj, (int, float, np.integer, np.float))

def isstring(obj): return istype(obj, str)

def isordered(obj):
    if not isnumeric(obj): raise ValueError('expected input to be numeric')
    if isiterable(obj):
        if isiterable_till(obj, 1):
            tmp = -float('inf')
            for i in obj:
                if i<tmp: return False
                else: tmp = i
            return True
        else: return all([isordered(i) for i in obj])
    else: return True

def get_color(vals, scale='linear', ax=None, ticks=None, format=None, label='', cmap='coolwarm', rotation='horizontal', **kwargs):
    from matplotlib import cm, colors, colorbar
    from matplotlib.ticker import MaxNLocator, LogLocator, LogFormatterSciNotation
    cmap = cm.get_cmap(cmap)
    vmin, vmax = min(vals), max(vals)
    if scale=='linear': norm = colors.Normalize(vmin=vmin, vmax=vmax)
    elif scale=='int':
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        if ticks is None: ticks = MaxNLocator(integer=True)
    elif 'log' in scale:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        if scale=='log2': base = 2
        else: base = 10
        if ticks is None: ticks = LogLocator(base=base)
        if format is None: format = LogFormatterSciNotation(base=base)
    if ax is not None:
        cax = colorbar.make_axes(ax, **kwargs)
        cbar = colorbar.ColorbarBase(ax=cax[0], cmap=cmap, norm=norm, ticks=ticks, format=format)
        if label: cbar.set_label(label, rotation=rotation)
        cbar.minorticks_off()
    return [cmap(norm(i)) for i in vals]

def get_legend(labels, colors=None, markers='', linestyles='-', markersize=None, lw=2, alpha=None, ax=None, **kwargs):
    from matplotlib.lines import Line2D
    if isinstance(labels, str): labels = [labels]
    if colors is None: colors = get_color(list(range(len(labels))))
    if markersize is None or isinstance(markersize, (int, float)): markersize = [markersize]*len(labels)
    if alpha is None or isinstance(alpha, (int, float)): alpha = [alpha]*len(labels)
    if isinstance(colors, str): colors = [colors]*len(labels)
    if isinstance(markers, str): markers = [markers]*len(labels)
    if isinstance(linestyles, str): linestyles = [linestyles]*len(labels)
    lines = [Line2D([0], [0], color=colors[i], lw=lw, marker=markers[i], markersize=markersize[i], alpha=alpha[i], ls=linestyles[i]) for i in range(len(labels))]
    if ax is not None: 
        ax.legend(lines, labels, **kwargs)
    else:
        return lines, labels

def get_marker(length):
    markers = ['$\\circ$', 'x','*', '^', 'o', '+', 'v', '.']
    l = len(markers)
    return [markers[i//l] for i in range(length)]

def get_axes(ax=None, figsize=(4, 3), dpi=180, equal=False, xscale='', yscale='', xlabel='', ylabel='', title='', **kwargs):
    import matplotlib.pyplot as plt
    if ax is None: fig, ax = plt.subplots(figsize=figsize, dpi=dpi, **kwargs)
    if equal: ax.set_aspect('equal')
    def set_axis(axis, scale):
        if scale=='int':
            from matplotlib.ticker import MaxNLocator
            if axis=='x': ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            elif axis=='y': ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        elif scale=='log2':
            if axis=='x': ax.set_xscale('log', base=2)
            elif axis=='y': ax.set_yscale('log', base=2)
        elif scale=='log10':
            if axis=='x': ax.set_xscale('log', base=10)
            elif axis=='y': ax.set_yscale('log', base=10)
    set_axis('x', xscale)
    set_axis('y', yscale)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    return ax

def insert_axes(ax, w='50%', h='50%', loc=4, pad=0.5, left=False, bottom=False, labelleft=False, labelbottom=False, **kwargs):
    if hasattr(ax, '__iter__'): return [insert_axes(i, w=w, h=h, loc=loc, pad=pad) for i in ax]
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_new = inset_axes(ax, width=w, height=h, loc=loc, borderpad=pad, **kwargs)
    ax_new.tick_params(left=left, bottom=bottom, labelleft=labelleft, labelbottom=labelbottom)#, right=True, top=True, labelright=True, labeltop=True)
    return ax_new

def draw_xy(ax, color='black', ls='--', **kwargs):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    vmin, vmax = min([xlim[0], ylim[0]]), max([xlim[1], ylim[1]])
    ax.plot((vmin, vmax), (vmin, vmax), color=color, ls=ls, **kwargs)
    ax.set_xlim((vmin, vmax))
    ax.set_ylim((vmin, vmax))

def draw_text(ax, text, loc='top', pad=0.05):
    if loc=='top': ax.set_title(text)
    elif loc=='top-left': ax.text(pad, 1-pad, text, transform=ax.transAxes, horizontalalignment='left', verticalalignment='top')
    elif loc=='top-right': ax.text(1-pad, 1-pad, text, transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')
    elif loc=='bottom-left': ax.text(pad, pad, text, transform=ax.transAxes, horizontalalignment='left', verticalalignment='bottom')
    elif loc=='bottom-right': ax.text(1-pad, pad, text, transform=ax.transAxes, horizontalalignment='right', verticalalignment='bottom')
    else: raise ValueError('unidentifiable text location "%s"'%str(loc))

def save_obj(filename, obj):
    from pickle import dump
    with open(filename, 'wb') as fd: dump(obj, fd)

def load_obj(filename):
    from pickle import load
    with open(filename, 'rb') as fd: return load(fd)

def time_foo(foo, *args, **kwargs):
    from datetime import datetime
    start = datetime.now()
    out = foo(*args, **kwargs)
    print(datetime.now()-start)
    return out
