import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from collections import deque


class SOM1D:
    def __init__(
        self,
        resolution=10,
        lr_x1=0.03,
        lr_x=0.001,
        neighborhood_decay = 3,
        conscience_factor=0.5,
        conscience_lr=0.03,
        prior_ema_alpha=0.001,
        input_min=None,
        input_max=None,
        visualize=False,
        viz_update_interval=1,
    ):
        self.resolution = resolution
        self.lr_x1 = lr_x1  # learning rate for x1 (conditional on instruction=1)
        self.lr_x = lr_x    # slow learning rate for marginal SOM (long memory)
        self.neighborhood_decay = neighborhood_decay
        self.conscience_factor = conscience_factor
        self.conscience_wf_learning_rate = conscience_lr
        self.prior_ema_alpha = prior_ema_alpha
        
        # Weights: if bounds are provided at construction, initialise via linspace;
        # otherwise start all neurons at 0 and let them adapt from observed data.
        if input_min is not None and input_max is not None:
            self.x_weights = np.linspace(input_min, input_max, resolution)
            self.x1_weights = np.linspace(input_min, input_max, resolution)
        else:
            self.x_weights = np.zeros(resolution)
            self.x1_weights = np.zeros(resolution)

        # Win frequencies for conscience mechanism
        self.x_winning_freq = np.full(resolution, 1.0 / resolution)
        self.x1_winning_freq = np.full(resolution, 1.0 / resolution)

        # Conscience biases
        self.x_conscience_bias = np.zeros(resolution)
        self.x1_conscience_bias = np.zeros(resolution)

        # Input space boundaries — expanded dynamically as observations arrive.
        # Both SOMs share the same boundaries so the voronoi_x / voronoi_x1 ratio
        # remains a valid density comparison across the same domain.
        # If bounds are provided at construction they are used immediately;
        # otherwise they remain None until the first call to step().
        self._input_min = float(input_min) if input_min is not None else None
        self._input_max = float(input_max) if input_max is not None else None

        # Voronoi cell sizes — initialised to 1.0; recomputed each step once bounds are known.
        self.voronoi_x  = np.ones(resolution)
        self.voronoi_x1 = np.ones(resolution)
        if self._input_min is not None:
            self._update_voronoi(self.x_weights,  self.voronoi_x)
            self._update_voronoi(self.x1_weights, self.voronoi_x1)
        self.inv_voronoi_x  = 1.0 / self.voronoi_x
        self.inv_voronoi_x1 = 1.0 / self.voronoi_x1

        self.prior_instruction = 0.5
        self.score_x1 = 0.0
        self.posterior_instruction = 0.5

        self.visualize = visualize
        self._viz_update_interval = viz_update_interval
        self._viz_step_count = 0
        if self.visualize:
            self._init_viz()

    def _find_cbmu(self, value, weights, conscience_bias):
        distances = np.abs(weights - value) - conscience_bias
        return np.argmin(distances)
    
    def _find_bmu(self, value, weights):
        distances = np.abs(weights - value)
        return np.argmin(distances)

    def _update_weights(self, bmu_idx, value, weights, lr=None):
        if lr is None:
            lr = self.lr
        weights[bmu_idx] += lr * (value - weights[bmu_idx])
        #update all other (non-BMU) neurons in the neighborhood where lr is scaled by e^(-dist/neighborhood_decay)
        for i in range(self.resolution):
            if i != bmu_idx:
                dist = abs(i - bmu_idx)
                influence = np.exp(-dist / self.neighborhood_decay)
                weights[i] += lr * influence * (value - weights[i])

    def _update_voronoi(self, weights, voronoi):
        # Guard: skip if bounds are unknown or the observed range is still degenerate.
        if self._input_min is None or (self._input_max - self._input_min) < 1e-12:
            return
        # Sort by weight value so we can correctly identify the two boundary neurons.
        # The neuron with the minimum weight owns everything from input_min to its
        # midpoint with the next neuron; the maximum-weight neuron owns its midpoint
        # to input_max.  Using index-order d[0]/d[-1] silently collapses these cells
        # to a single inter-neuron gap once the SOM compresses into a tight cluster,
        # making the score formula see spuriously high conditional density everywhere.
        sorted_idx = np.argsort(weights)
        sw = weights[sorted_idx]
        d = np.abs(np.diff(sw))
        sv = np.empty(len(weights))
        sv[0]    = (sw[0] - self._input_min) + d[0] / 2
        sv[1:-1] = (d[:-1] + d[1:]) / 2
        sv[-1]   = (self._input_max - sw[-1]) + d[-1] / 2
        voronoi[sorted_idx] = sv

    def _update_conscience(self, bmu_idx, winning_freq, conscience_bias):
        # Update winning frequency for BMU
        winning_freq[bmu_idx] += self.conscience_wf_learning_rate * (1.0 - winning_freq[bmu_idx])
        # Decay winning frequency for non-BMUs
        for i in range(self.resolution):
            if i != bmu_idx:
                winning_freq[i] += self.conscience_wf_learning_rate * (0.0 - winning_freq[i])
        # Update conscience bias based on new winning frequencies
        conscience_bias[:] = self.conscience_factor * ((1.0/self.resolution) - winning_freq)

    def step(self, observation, instruction):
        # Expand the observed input range to cover this observation.
        # Both SOMs use the same boundaries so the voronoi_x / voronoi_x1
        # density ratio remains meaningful across the same domain.
        if self._input_min is None:
            self._input_min = float(observation)
            self._input_max = float(observation)
        else:
            if observation < self._input_min:
                self._input_min = float(observation)
            if observation > self._input_max:
                self._input_max = float(observation)

        # Step 1: Find BMU in x_weights
        bmu_x = self._find_cbmu(observation, self.x_weights, self.x_conscience_bias)

        # Step 2: Update weights
        self._update_weights(bmu_x, observation, self.x_weights, lr=self.lr_x)

        # Step 3: Update voronoi_x
        self._update_voronoi(self.x_weights, self.voronoi_x)

        # Step 4: Update conscience for x
        self._update_conscience(bmu_x, self.x_winning_freq, self.x_conscience_bias)

        # Step 5.1: Find BMU for conditional SOM.
        # Two separate lookups:
        #   bmu_x1_score — pure nearest-neighbour, no conscience bias.  Used only
        #                  for density scoring in Step 8.  Conscience bias is a
        #                  training aid; injecting it here corrupts the density
        #                  estimate, especially for edge neurons whose x1 firing
        #                  frequency is under-counted (instruction=0 wins are never
        #                  tracked, so conscience over-boosts them).
        #   bmu_x1_train — conscience-adjusted BMU, used only to drive weight and
        #                  conscience updates when instruction == 1.
        bmu_x1_score = self._find_bmu(observation, self.x1_weights)

        lr_x1 = self.lr_x1

        # Update x1 SOM only when instruction == 1.0
        if instruction == 1.0:
            bmu_x1_train = self._find_cbmu(observation, self.x1_weights, self.x1_conscience_bias)

            # Step 5a.2: Update weights
            self._update_weights(bmu_x1_train, observation, self.x1_weights, lr=lr_x1)

            # Step 5a.3: Update voronoi_x1 
            self._update_voronoi(self.x1_weights, self.voronoi_x1)

            # Step 5a.4: Update conscience for x1
            self._update_conscience(bmu_x1_train, self.x1_winning_freq, self.x1_conscience_bias)

        # Step 6: Calculate the inverse Voronoi cell size (1 / voronoi) for x, x1
        self.inv_voronoi_x = np.where(self.voronoi_x != 0, 1.0 / np.where(self.voronoi_x != 0, self.voronoi_x, 1.0), 0.0)
        self.inv_voronoi_x1 = np.where(self.voronoi_x1 != 0, 1.0 / np.where(self.voronoi_x1 != 0, self.voronoi_x1, 1.0), 0.0)

        # Step 7: Update EMA estimate of P(instruction = 1)
        self.prior_instruction += self.prior_ema_alpha * (float(instruction) - self.prior_instruction)

        # Step 8: Calculate and X1 scores for this observation
        self.score_x1 = self.prior_instruction * (self.voronoi_x[bmu_x] / self.voronoi_x1[bmu_x1_score]) if self.voronoi_x1[bmu_x1_score] != 0 else 0.0
        self.posterior_instruction = 1.0 if self.score_x1 >= 0.5 else 0.0

        if self.visualize:
            self._viz_step_count += 1
            if self._viz_step_count % self._viz_update_interval == 0:
                self._update_viz(instruction)

        # Step 9: Return predicted instruction
        return self.score_x1, self.posterior_instruction

    def _init_viz(self):
        self._ts_instruction = deque(maxlen=200)
        self._ts_score_x1 = deque(maxlen=200)
        self._ts_posterior = deque(maxlen=200)
        self._ts_prior = deque(maxlen=200)

        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        pg.setConfigOptions(antialias=True)
        self._win = pg.GraphicsLayoutWidget(title='SOM1D Dashboard')
        self._win.resize(1200, 1200)

        x_idx = np.arange(self.resolution, dtype=float)
        bar_w = 0.7
        colors = [(180, 130, 70), (80, 127, 255)]
        col_titles = ['x (Marginal)', 'x1 (instruction=1)']
        weights_init = [self.x_weights, self.x1_weights]
        freq_init = [self.x_winning_freq, self.x1_winning_freq]

        self._weight_curves = []
        self._freq_bars = []
        self._plots_freq = []
        self._plots_weights = []

        for col in range(2):
            r, g, b = colors[col]
            pen = pg.mkPen(color=(r, g, b), width=2)
            brush = pg.mkBrush(r, g, b, 200)

            # Row 0: weights (line + markers)
            p = self._win.addPlot(row=0, col=col)
            p.setTitle(f'{col_titles[col]} — Weights')
            p.setLabel('bottom', 'Neuron')
            p.showGrid(y=True, alpha=0.3)
            p.enableAutoRange('y')
            c = p.plot(x_idx, weights_init[col].copy(), pen=pen,
                       symbol='o', symbolSize=6, symbolBrush=brush, symbolPen=None)
            self._weight_curves.append(c)
            self._plots_weights.append(p)

            # Row 1: winning frequency (bar chart)
            p = self._win.addPlot(row=1, col=col)
            p.setTitle(f'{col_titles[col]} — Winning Freq')
            p.setLabel('bottom', 'Neuron')
            p.showGrid(y=True, alpha=0.3)
            p.enableAutoRange('y')
            bar = pg.BarGraphItem(x=x_idx, height=freq_init[col].copy(),
                                  width=bar_w, brush=brush, pen=pg.mkPen(None))
            p.addItem(bar)
            self._freq_bars.append(bar)
            self._plots_freq.append(p)

        # Rows 2-5: four stacked time series spanning both columns
        ts_specs = [
            ('Instruction',                  (200, 200, 200), False),
            ('Score x1',                     ( 80, 200, 120), False),
            ('Posterior P(instruction=1)',   ( 80, 127, 255), True),
            ('Prior P(instruction=1)',       (220,  80, 220), True),
        ]
        self._ts_plots = []
        self._ts_curves = []
        for i, (title, color, fixed_range) in enumerate(ts_specs):
            p = self._win.addPlot(row=2 + i, col=0, colspan=2)
            p.setTitle(title)
            p.setLabel('bottom', 'Step')
            p.showGrid(y=True, alpha=0.3)
            if fixed_range:
                p.setYRange(0.0, 1.0)
                p.addLine(y=0.5, pen=pg.mkPen(color=(150, 150, 150), width=1,
                          style=pg.QtCore.Qt.PenStyle.DashLine))
            else:
                p.enableAutoRange('y')
            c = p.plot([], pen=pg.mkPen(color=color, width=1))
            self._ts_curves.append(c)
            self._ts_plots.append(p)

        self._win.show()

    def _update_viz(self, instruction):
        self._ts_instruction.append(float(instruction))
        self._ts_score_x1.append(self.score_x1)
        self._ts_posterior.append(self.posterior_instruction)
        self._ts_prior.append(self.prior_instruction)

        x_idx = np.arange(self.resolution, dtype=float)
        weights = [self.x_weights, self.x1_weights]
        freqs = [self.x_winning_freq, self.x1_winning_freq]

        for col in range(2):
            self._weight_curves[col].setData(x_idx, weights[col])
            cur_max_w = float(np.max(weights[col]))
            cur_min_w = float(np.min(weights[col]))
            pad = (cur_max_w - cur_min_w) * 0.1 or 0.1
            self._plots_weights[col].setYRange(cur_min_w - pad, cur_max_w + pad)

            self._freq_bars[col].setOpts(height=freqs[col])
            cur_max_f = float(np.max(freqs[col]))
            cur_min_f = float(np.min(freqs[col]))
            pad = (cur_max_f - cur_min_f) * 0.1 or 0.1
            self._plots_freq[col].setYRange(cur_min_f - pad, cur_max_f + pad)

        ts_data = [self._ts_instruction, self._ts_score_x1, self._ts_posterior, self._ts_prior]
        for i, ts_deque in enumerate(ts_data):
            ts = np.array(ts_deque)
            self._ts_curves[i].setData(np.arange(len(ts), dtype=float), ts)
            if len(ts) > 0 and i < 2:  # auto-range only for instruction and score_x1
                cur_max_ts = float(np.max(ts))
                cur_min_ts = float(np.min(ts))
                pad = (cur_max_ts - cur_min_ts) * 0.1 or 0.1
                self._ts_plots[i].setYRange(cur_min_ts - pad, cur_max_ts + pad)

        self._app.processEvents()
