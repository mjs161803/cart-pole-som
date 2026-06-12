import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from collections import deque

class BayesianNeuron:
    def __init__(
        self,
        resolution=10,
        lr = 0.001,
        conscience_lr = 0.001,
        conscience_factor = 0.5,
        prior_ema_alpha = 0.001,
        neighborhood_decay = 3,
        obs_min=None,
        obs_max=None,
        visualize=False,
        viz_update_interval=100,
        feature_name='',
    ):
        self.resolution = resolution
        self.equiprobability = 1.0 / resolution
        self.lr = lr
        self.conscience_lr = conscience_lr
        self.conscience_factor = conscience_factor
        self.prior_ema_alpha = prior_ema_alpha
        self.neighborhood_decay = neighborhood_decay
        self.visualize = visualize
        self.viz_update_interval = viz_update_interval
        self.feature_name = feature_name
        # Initialise weights as linspace when bounds are known so all neurons
        # start spread across the input space and none are stranded out-of-range.
        # Without known bounds, initialise to zeros so they spread from the data.
        if obs_min is not None and obs_max is not None:
            self.w_m = np.linspace(obs_min, obs_max, resolution)
            self.w_c = np.linspace(obs_min, obs_max, resolution)
        else:
            self.w_m = np.zeros(resolution)
            self.w_c = np.zeros(resolution)
        self.obs_min = float(obs_min) if obs_min is not None else None
        self.obs_max = float(obs_max) if obs_max is not None else None
        self.v_m = np.zeros(resolution)     # Voronoi regions for marginal neuron array
        self.v_c = np.zeros(resolution)     # Voronoi regions for conditional neuron array
        self.b_m = np.zeros(resolution)     # Conscience bias for marginal neuron array
        self.b_c = np.zeros(resolution)     # Conscience bias for conditional neuron array
        self.p_m = np.full(resolution, 1.0/resolution)  # Win percentage for marginal neuron array
        self.p_c = np.full(resolution, 1.0/resolution)  # Win percentage for conditional neuron array
        self.y_m = np.zeros(resolution)     # Win state of marginal neuron array (1 if won, 0 otherwise)
        self.y_c = np.zeros(resolution)     # Win state of conditional neuron array (1 if won, 0 otherwise)
        self.prior = 0.0                    # Prior probability P(instruction = 1)
        self.posterior = 0.0                # Posterior probability P(instruction = 1 | observation)
        self.prediction = 0.0               # Final prediction after thresholding posterior
        self._viz_step_count = 0
        if self.visualize:
            self._init_viz()

    def step(self, observation, instruction):
        #1) Update self.obs_min and self.obs_max based on the new observation
        self.update_minmax(observation)

        #2) Update self.prior using an EMA of the instruction
        self.update_prior(instruction)

        #3) Find BMU index of marginal neuron array using conscience-biased distance
        bmu_cb_m = self.find_cbmu_marginal(observation)

        #4) Update conscience bias of marginal neurons
        self.update_conscience_marginal()

        #5) Update weight of winning marginal neuron and resort (w_m, b_m, p_m) arrays
        self.update_weight_marginal(bmu_cb_m, observation)

        #6) Update Voronoi region of winning marginal neuron and nearest neighbors
        self.update_voronoi_marginal()

        #7) If instruction == 1, Update conditional neuron array using conscience-biased distance, weight update, and Voronoi update:
        if instruction == 1:
            #7.1) Find BMU index of conditional neuron array using conscience-biased distance
            bmu_cb_c = self.find_cbmu_conditional(observation)
            #7.2) Update conscience bias of conditional neurons
            self.update_conscience_conditional()
            #7.3) Update weight of winning conditional neuron and resort (w_c, b_c, p_c) arrays
            self.update_weight_conditional(bmu_cb_c, observation)
            #7.4) Update Voronoi region of winning conditional neuron and nearest neighbors
            self.update_voronoi_conditional()

        #8) Find BMU index of marginal neuron array using standard distance
        bmu_m = self.find_bmu_marginal(observation)

        #9) Find BMU index of conditional neuron array using standard distance
        bmu_c = self.find_bmu_conditional(observation)

        #10) Calculate posterior probability P(instruction = 1 | observation) using Bayes' theorem
        self.calc_posterior(bmu_m, bmu_c)

        #11) Calculate final prediction by thresholding posterior at 0.5
        self.calc_prediction()

        #12) Calculate pointwise mutual information between observation and instruction using marginal and conditional BMUs
        pmi = self.calc_pmi(bmu_m, bmu_c)
    
        if self.visualize:
            self._viz_step_count += 1
            if self._viz_step_count % self.viz_update_interval == 0:
                self._update_viz(instruction)

        return self.prediction, self.posterior, pmi
    
    def update_minmax(self, observation):
        if self.obs_min is None:
            self.obs_min = float(observation)
        else:
            self.obs_min = min(self.obs_min, float(observation))
        if self.obs_max is None:
            self.obs_max = float(observation)
        else:
            self.obs_max = max(self.obs_max, float(observation))
        return
    
    def update_prior(self, instruction):
        self.prior = self.prior_ema_alpha * instruction + (1 - self.prior_ema_alpha) * self.prior
        return
    
    def find_cbmu_marginal(self, observation):
        d = np.abs(self.w_m - observation) - self.b_m  # Conscience-biased distance
        cbmu_index = np.argmin(d)
        # Update win states for marginal neurons
        self.y_m = np.zeros(self.resolution)
        self.y_m[cbmu_index] = 1
        return cbmu_index
    
    def update_conscience_marginal(self):
        #1) Update marginal neurons' win percentages using current win states
        self.p_m = self.p_m + self.conscience_lr* (self.y_m - self.p_m)
        self.b_m = self.conscience_factor * (self.equiprobability - self.p_m)
        return    
    
    def update_weight_marginal(self, bmu_index, observation):
        #1) Apply neighbourhood update: all neurons move toward observation,
        #   scaled by a Gaussian centred on the BMU.  This pulls in dead neurons.
        indices = np.arange(self.resolution)
        influence = np.exp(-np.abs(indices - bmu_index) / self.neighborhood_decay)
        self.w_m += self.lr * influence * (observation - self.w_m)
        #2) Resort (w_m, b_m, p_m) arrays based on updated weights while keeping them aligned
        sorted_indices = np.argsort(self.w_m)
        self.w_m = self.w_m[sorted_indices]
        self.b_m = self.b_m[sorted_indices]
        self.p_m = self.p_m[sorted_indices]        
        return
    
    def update_voronoi_marginal(self):
        # For each marginal neuron, calculate its Voronoi region as the midpoint between its weight and the weights of its neighbors
        #   If it's the first neuron, its Voronoi region extends from self.min to the midpoint with the next neuron
        #   If it's the last neuron, its Voronoi region extends from the midpoint with the previous neuron to self.max
        #   Otherwise, its Voronoi region extends from the midpoint with the previous neuron to the midpoint with the next neuron
        for i in range(self.resolution):
            if i == 0:
                self.v_m[i] = (self.w_m[0] + self.w_m[1]) / 2.0 - self.obs_min
            elif i == self.resolution - 1:
                self.v_m[i] = self.obs_max - (self.w_m[-2] + self.w_m[-1]) / 2.0
            else:
                self.v_m[i] = (self.w_m[i+1] - self.w_m[i-1]) / 2.0
        return

    
    def find_cbmu_conditional(self, observation):
        d = np.abs(self.w_c - observation) - self.b_c  # Conscience-biased distance
        cbmu_index = np.argmin(d)
        # Update win states for conditional neurons
        self.y_c = np.zeros(self.resolution)
        self.y_c[cbmu_index] = 1
        return cbmu_index
    
    def update_conscience_conditional(self):
        #1) Update conditional neurons' win percentages using current win states
        self.p_c = self.p_c + self.conscience_lr* (self.y_c - self.p_c)
        self.b_c = self.conscience_factor * (self.equiprobability - self.p_c)
        return    
    
    def update_weight_conditional(self, bmu_index, observation):
        #1) Apply neighbourhood update: all neurons move toward observation,
        #   scaled by a Gaussian centred on the BMU.  This pulls in dead neurons.
        indices = np.arange(self.resolution)
        influence = np.exp(-np.abs(indices - bmu_index) / self.neighborhood_decay)
        self.w_c += self.lr * influence * (observation - self.w_c)
        #2) Resort (w_c, b_c, p_c) arrays based on updated weights while keeping them aligned
        sorted_indices = np.argsort(self.w_c)
        self.w_c = self.w_c[sorted_indices]
        self.b_c = self.b_c[sorted_indices]
        self.p_c = self.p_c[sorted_indices]        
        return
    
    def update_voronoi_conditional(self):
        # For each conditional neuron, calculate its Voronoi region as the midpoint between its weight and the weights of its neighbors
        #   If it's the first neuron, its Voronoi region extends from self.min to the midpoint with the next neuron
        #   If it's the last neuron, its Voronoi region extends from the midpoint with the previous neuron to self.max
        #   Otherwise, its Voronoi region extends from the midpoint with the previous neuron to the midpoint with the next neuron
        for i in range(self.resolution):
            if i == 0:
                self.v_c[i] = (self.w_c[0] + self.w_c[1]) / 2.0 - self.obs_min
            elif i == self.resolution - 1:
                self.v_c[i] = self.obs_max - (self.w_c[-2] + self.w_c[-1]) / 2.0
            else:
                self.v_c[i] = (self.w_c[i+1] - self.w_c[i-1]) / 2.0
        return
    
    def find_bmu_marginal(self, observation):
        idx = np.searchsorted(self.w_m, observation)
        if idx == 0:
            bmu_index = 0
        elif idx == self.resolution:
            bmu_index = self.resolution - 1
        else:
            if abs(observation - self.w_m[idx-1]) <= abs(observation - self.w_m[idx]):
                bmu_index = idx - 1
            else:
                bmu_index = idx
        return bmu_index
    
    def find_bmu_conditional(self, observation):
        idx = np.searchsorted(self.w_c, observation)
        if idx == 0:
            bmu_index = 0
        elif idx == self.resolution:
            bmu_index = self.resolution - 1
        else:            
            if abs(observation - self.w_c[idx-1]) <= abs(observation - self.w_c[idx]):
                bmu_index = idx - 1
            else:
                bmu_index = idx
        return bmu_index
    
    def calc_posterior(self, bmu_m, bmu_c):
        if self.v_c[bmu_c] <= 0.0:
            self.posterior = 0.0
        else:
            self.posterior = self.prior * (self.v_m[bmu_m] / self.v_c[bmu_c])  # P(instruction=1|observation) = P(instruction=1) * P(observation|instruction=1) / P(observation)
        return
    
    def calc_prediction(self):
        self.prediction = 1 if self.posterior >= 0.5 else 0
        return
    
    def calc_pmi(self, bmu_m, bmu_c):
        if self.v_c[bmu_c] <= 0.0 or self.v_m[bmu_m] <= 0.0:
            return 0.0
        pmi = np.log2(self.v_c[bmu_c]/self.v_m[bmu_m])  # PMI(observation; instruction) = log2(P(observation|instruction=1) / P(observation))
        return pmi

    def _init_viz(self):
        self._ts_instruction = deque(maxlen=200)
        self._ts_prediction = deque(maxlen=200)
        self._ts_posterior = deque(maxlen=200)
        self._ts_prior = deque(maxlen=200)

        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        pg.setConfigOptions(antialias=True)
        _title = self.feature_name if self.feature_name else 'BayesianNeuron Dashboard'
        self._win = pg.GraphicsLayoutWidget(title=_title)
        self._win.resize(1200, 1100)

        x_idx = np.arange(self.resolution, dtype=float)
        bar_w = 0.7
        color_m = (180, 130, 70)
        color_c = (80, 127, 255)

        self._weight_curves = []
        self._plots_weights = []
        self._voronoi_bars = []
        self._plots_voronoi = []
        self._winpct_bars = []
        self._plots_winpct = []
        self._bias_bars = []
        self._plots_bias = []

        for col, (title, color, data) in enumerate([
            ('w_m (Marginal Weights)', color_m, self.w_m),
            ('w_c (Conditional Weights)', color_c, self.w_c),
        ]):
            r, g, b = color
            pen = pg.mkPen(color=(r, g, b), width=2)
            brush = pg.mkBrush(r, g, b, 200)
            p = self._win.addPlot(row=0, col=col)
            p.setTitle(title)
            p.setLabel('bottom', 'Neuron')
            p.showGrid(y=True, alpha=0.3)
            p.enableAutoRange('y')
            c = p.plot(x_idx, data.copy(), pen=pen,
                       symbol='o', symbolSize=6, symbolBrush=brush, symbolPen=None)
            self._weight_curves.append(c)
            self._plots_weights.append(p)

        for col, (title, color, data) in enumerate([
            ('v_m (Marginal Voronoi)', color_m, self.v_m),
            ('v_c (Conditional Voronoi)', color_c, self.v_c),
        ]):
            r, g, b = color
            brush = pg.mkBrush(r, g, b, 200)
            p = self._win.addPlot(row=1, col=col)
            p.setTitle(title)
            p.setLabel('bottom', 'Neuron')
            p.showGrid(y=True, alpha=0.3)
            p.enableAutoRange('y')
            bar = pg.BarGraphItem(x=x_idx, height=data.copy(), width=bar_w, brush=brush, pen=pg.mkPen(None))
            p.addItem(bar)
            self._voronoi_bars.append(bar)
            self._plots_voronoi.append(p)

        for col, (title, color, data) in enumerate([
            ('p_m (Marginal Win %)', color_m, self.p_m),
            ('p_c (Conditional Win %)', color_c, self.p_c),
        ]):
            r, g, b = color
            brush = pg.mkBrush(r, g, b, 200)
            p = self._win.addPlot(row=2, col=col)
            p.setTitle(title)
            p.setLabel('bottom', 'Neuron')
            p.showGrid(y=True, alpha=0.3)
            p.enableAutoRange('y')
            bar = pg.BarGraphItem(x=x_idx, height=data.copy(), width=bar_w, brush=brush, pen=pg.mkPen(None))
            p.addItem(bar)
            self._winpct_bars.append(bar)
            self._plots_winpct.append(p)

        for col, (title, color, data) in enumerate([
            ('b_m (Marginal Conscience Bias)', color_m, self.b_m),
            ('b_c (Conditional Conscience Bias)', color_c, self.b_c),
        ]):
            r, g, b = color
            brush = pg.mkBrush(r, g, b, 200)
            p = self._win.addPlot(row=3, col=col)
            p.setTitle(title)
            p.setLabel('bottom', 'Neuron')
            p.showGrid(y=True, alpha=0.3)
            p.enableAutoRange('y')
            p.addLine(y=0.0, pen=pg.mkPen(color=(150, 150, 150), width=1,
                      style=pg.QtCore.Qt.PenStyle.DashLine))
            bar = pg.BarGraphItem(x=x_idx, height=data.copy(), width=bar_w, brush=brush, pen=pg.mkPen(None))
            p.addItem(bar)
            self._bias_bars.append(bar)
            self._plots_bias.append(p)

        # Row 4: Prior time series
        p = self._win.addPlot(row=4, col=0, colspan=2)
        p.setTitle('Prior P(instruction=1)')
        p.setLabel('bottom', 'Step')
        p.showGrid(y=True, alpha=0.3)
        p.setYRange(0.0, 1.0)
        p.addLine(y=0.5, pen=pg.mkPen(color=(150, 150, 150), width=1,
                  style=pg.QtCore.Qt.PenStyle.DashLine))
        self._prior_curve = p.plot([], pen=pg.mkPen(color=(220, 80, 220), width=1))
        self._prior_plot = p

        # Row 5: Instruction / Prediction / Posterior combined
        p = self._win.addPlot(row=5, col=0, colspan=2)
        _lo = f'{self.obs_min:.3f}' if self.obs_min is not None else 'n/a'
        _hi = f'{self.obs_max:.3f}' if self.obs_max is not None else 'n/a'
        p.setTitle(
            f'Instruction / Prediction / Posterior  |  '
            f'obs_min={_lo}  obs_max={_hi}'
        )
        p.setLabel('bottom', 'Step')
        p.showGrid(y=True, alpha=0.3)
        p.setYRange(-0.1, 1.1)
        p.addLine(y=0.5, pen=pg.mkPen(color=(150, 150, 150), width=1,
                  style=pg.QtCore.Qt.PenStyle.DashLine))
        self._instr_curve = p.plot([], pen=pg.mkPen(color=(200, 200, 200), width=1), name='Instruction')
        self._pred_curve  = p.plot([], pen=pg.mkPen(color=(80, 200, 120), width=1), name='Prediction')
        self._post_curve  = p.plot([], pen=pg.mkPen(color=(80, 127, 255), width=1), name='Posterior')
        self._combined_plot = p

        self._win.show()

    def _update_viz(self, instruction):
        self._ts_instruction.append(float(instruction))
        self._ts_prediction.append(float(self.prediction))
        self._ts_posterior.append(float(self.posterior))
        self._ts_prior.append(float(self.prior))

        x_idx = np.arange(self.resolution, dtype=float)

        for col, weights in enumerate([self.w_m, self.w_c]):
            weights_safe = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            self._weight_curves[col].setData(x_idx, weights_safe)
            finite_vals = weights_safe[np.isfinite(weights_safe)]
            if finite_vals.size > 0:
                cur_max = float(np.max(finite_vals))
                cur_min = float(np.min(finite_vals))
                pad = (cur_max - cur_min) * 0.1 or 0.1
                self._plots_weights[col].setYRange(cur_min - pad, cur_max + pad)

        for col, voronoi in enumerate([self.v_m, self.v_c]):
            voronoi_safe = np.nan_to_num(voronoi, nan=0.0, posinf=0.0, neginf=0.0)
            self._voronoi_bars[col].setOpts(height=voronoi_safe)
            finite_vals = voronoi_safe[np.isfinite(voronoi_safe)]
            if finite_vals.size > 0:
                cur_max = float(np.max(finite_vals))
                cur_min = float(np.min(finite_vals))
                pad = (cur_max - cur_min) * 0.1 or 0.01
                self._plots_voronoi[col].setYRange(max(0.0, cur_min - pad), cur_max + pad)

        for col, winpct in enumerate([self.p_m, self.p_c]):
            winpct_safe = np.nan_to_num(winpct, nan=0.0, posinf=0.0, neginf=0.0)
            self._winpct_bars[col].setOpts(height=winpct_safe)
            finite_vals = winpct_safe[np.isfinite(winpct_safe)]
            if finite_vals.size > 0:
                cur_max = float(np.max(finite_vals))
                cur_min = float(np.min(finite_vals))
                pad = (cur_max - cur_min) * 0.1 or 0.001
                self._plots_winpct[col].setYRange(max(0.0, cur_min - pad), cur_max + pad)

        for col, bias in enumerate([self.b_m, self.b_c]):
            bias_safe = np.nan_to_num(bias, nan=0.0, posinf=0.0, neginf=0.0)
            self._bias_bars[col].setOpts(height=bias_safe)
            finite_vals = bias_safe[np.isfinite(bias_safe)]
            if finite_vals.size > 0:
                cur_max = float(np.max(finite_vals))
                cur_min = float(np.min(finite_vals))
                pad = (cur_max - cur_min) * 0.1 or 0.01
                self._plots_bias[col].setYRange(cur_min - pad, cur_max + pad)

        ts_prior = np.nan_to_num(np.array(self._ts_prior), nan=0.0, posinf=0.0, neginf=0.0)
        self._prior_curve.setData(np.arange(len(ts_prior), dtype=float), ts_prior)

        n = len(self._ts_instruction)
        t = np.arange(n, dtype=float)
        self._instr_curve.setData(t, np.nan_to_num(np.array(self._ts_instruction), nan=0.0, posinf=0.0, neginf=0.0))
        self._pred_curve.setData(t, np.nan_to_num(np.array(self._ts_prediction), nan=0.0, posinf=0.0, neginf=0.0))
        self._post_curve.setData(t, np.nan_to_num(np.array(self._ts_posterior), nan=0.0, posinf=0.0, neginf=0.0))
        _lo = f'{self.obs_min:.3f}' if self.obs_min is not None else 'n/a'
        _hi = f'{self.obs_max:.3f}' if self.obs_max is not None else 'n/a'
        self._combined_plot.setTitle(
            f'Instruction / Prediction / Posterior  |  '
            f'obs_min={_lo}  obs_max={_hi}'
        )

        self._app.processEvents()
    


                