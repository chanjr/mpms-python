#!/usr/bin/env python3

import lmfit
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np

class VarSlider:
    def __init__(self, ax, valname, valmin, valmax,
            valinit = None, valstep = None, callback = None):
        self.ax = ax
        self.valname = valname
        if valinit is None:
            valinit = (valmax + valmin)/2
        self.slider = Slider(self.ax, valname, valmin, valmax,
                valinit = valinit, valstep = valstep)
        if callback is not None:
            self.cid = self.slider.on_changed(callback)
        else:
            self.cid = None

    @property
    def valmin(self):
        return self.slider.valmin

    @valmin.setter
    def valmin(self, v):
        self.slider.valmin = v
        self.ax.set_xlim(left = v)

    @property
    def valmax(self):
        return self.slider.valmax

    @valmax.setter
    def valmax(self, v):
        self.slider.valmax = v
        self.ax.set_xlim(right = v)

    @property
    def valstep(self):
        return self.slider.valstep

    @valstep.setter
    def valstep(self, v):
        self.slider.valstep = v

    @property
    def value(self):
        return self.slider.val

    @value.setter
    def value(self, v):
        self.slider.set_val(v)

class Ilmfit:
    def __init__(self, model, params = None, data = None, bounds = None, x =
            None, y = None):
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.fig.set_size_inches(8, 5)
        plt.tight_layout()
        self.fig.subplots_adjust(right = 0.55)

        self.bounds = bounds
        self.data = data
        if data is not None:
            self.x, self.y = data.T
            self.data_line = self.ax.plot(self.x, self.y, label = 'Data',
                    linestyle = '--', marker = '.')
        elif x is not None:
            self.x = x
            if y is not None:
                self.y = y
                self.data_line = self.ax.plot(self.x, self.y, label = 'Data',
                        linestyle = '--', marker = '.')
            else:
                self.y = None
                self.data_line = None
        else:
            self.x = np.linspace(-5, 5, 100)
            self.y = None
            self.data_line = None

        self.model = model
        self.model_lines = []
        if len(self.model.components) > 1:
            self.model_lines += [self.ax.plot([], [], label = c.name)[0] for c in
                    self.model.components]
        self.model_lines.append(self.ax.plot([], [], label = 'Total')[0])
        if params is not None:
            self.params = params
        else:
            self.params = model.make_params()
        self.sliders = self.make_sliders()
        self.ax.legend()
        self.autoscale = True
        self.update()

    def make_sliders(self):
        params = [v for v in self.params.values() if not v.expr]
        sliders = {}
        ys = np.linspace(0.9, 0.1, len(params) + 1)
        for p, y in zip(params, ys):
            ax = self.fig.add_axes([0.65, y, 0.20, 0.03])
            try:
                valmin, valmax = self.bounds.get(p.name)
            except:
                valmin = p.min if ~np.isinf(p.min) else -1
                valmax = p.max if ~np.isinf(p.max) else 1
            valinit = p.value if ~np.isinf(p.value) else (valmin + valmax)/2
            sliders[p.name] = VarSlider(ax, p.name, valmin, valmax,
                    valinit = valinit, callback = self.update)
        button_ax = self.fig.add_axes([0.65, 0.1, 0.20, 0.09])
        self.button = Button(button_ax, 'Fit')
        self.button.on_clicked(self.fit)

        return sliders

    @property
    def independent_var(self):
        return {self.model.independent_vars[0] : self.x}

    def update(self, *args, **kwargs):
        for k, p in self.sliders.items():
            self.params[k].set(value = p.value)
        if len(self.model.components) > 1:
            for c, l in zip(self.model.components, self.model_lines):
                l.set_data(self.x, c.eval(params = self.params,
                    **self.independent_var))
        self.model_lines[-1].set_data(self.x, self.model.eval(params =
            self.params, **self.independent_var))
        # autoscale
        if self.autoscale:
            self.ax.relim()
            self.ax.autoscale_view()
        pass

    def fit(self, *args, **kwargs):
        self.res = self.model.fit(self.y, params = self.params,
                **self.independent_var, **kwargs)
        for p, v in self.res.best_values.items():
            self.sliders[p].value = v
        self.update()

def plot_res(res):
    # component plotting for composite models
    fig, (ax, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios': [3, 1]})
    x = res.userkws[res.model.independent_vars[0]]
    y = res.data
    ax.plot(x, y, linestyle = '', marker = '.', label = 'Raw')
    for m in res.components:
        my = m.eval(params = res.params, **res.userkws)
        ax.plot(x, my, label = m.name)
    if len(res.components) > 1:
        ax.plot(x, res.eval(**res.userkws), linestyle = '--', label = 'Total Fit')
    ax2.plot(x, res.eval(**res.userkws) - y, marker = '.', linestyle = '', label = 'Residual')
    ax.legend()
    ax2.legend()
    plt.tight_layout()
