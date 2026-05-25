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
        lr_x0=0.001,
        neighborhood_decay = 3,
        conscience_factor=0.5,
        conscience_lr=0.03,
        visualize=False,
        viz_update_interval=1,
    ):
        self.resolution = resolution
        self.lr_x1 = lr_x1  # learning rate for x1 (conditional on instruction=1)
        self.lr_x = lr_x    # slow learning rate for marginal SOM (long memory)
        self.lr_x0 = lr_x0  # slow learning rate for x0 SOM (long memory)
        self.neighborhood_decay = neighborhood_decay
        self.conscience_factor = conscience_factor
        self.conscience_wf_learning_rate = conscience_lr
        
        # Weights: monotonically increasing, equally spaced
        self.x_weights = np.linspace(-np.pi, np.pi, resolution)
        self.x1_weights = np.linspace(-np.pi, np.pi, resolution)
        self.x0_weights = np.linspace(-np.pi, np.pi, resolution)

        # Win frequencies for conscience mechanism
        self.x_winning_freq = np.full(resolution, 1.0 / resolution)
        self.x1_winning_freq = np.full(resolution, 1.0 / resolution)
        self.x0_winning_freq = np.full(resolution, 1.0 / resolution)

        # Conscience biases
        self.x_conscience_bias = np.zeros(resolution)
        self.x1_conscience_bias = np.zeros(resolution)
        self.x0_conscience_bias = np.zeros(resolution)

        # Average radius to immediate topological neighbors
        def _voronoi_from_weights(w):
            d = np.diff(w)
            ar = np.empty(len(w))
            ar[0] = d[0]
            ar[1:-1] = (d[:-1] + d[1:]) / 2
            ar[-1] = d[-1]
            return ar

        self.voronoi_x = _voronoi_from_weights(self.x_weights)
        self.voronoi_x1 = _voronoi_from_weights(self.x1_weights)
        self.voronoi_x0 = _voronoi_from_weights(self.x0_weights)
        self.inv_voronoi_x = 1 / self.voronoi_x
        self.inv_voronoi_x1 = 1 / self.voronoi_x1
        self.inv_voronoi_x0 = 1 / self.voronoi_x0
        self.prior_instruction = 0.5
        self.score_x1 = 0.0
        self.score_x0 = 0.0

        self.visualize = visualize
        self._viz_update_interval = viz_update_interval
        self._viz_step_count = 0
        if self.visualize:
            self._init_viz()

    def _find_bmu(self, value, weights, conscience_bias):
        distances = np.abs(weights - value) - conscience_bias
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
        d = np.abs(np.diff(weights))
        voronoi[0] = d[0]
        voronoi[1:-1] = (d[:-1] + d[1:]) / 2
        voronoi[-1] = d[-1]

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
        # Step 1: Find BMU in x_weights
        bmu_x = self._find_bmu(observation, self.x_weights, self.x_conscience_bias)

        # Step 2: Update weights
        self._update_weights(bmu_x, observation, self.x_weights, lr=self.lr_x)

        # Step 3: Update voronoi_x
        self._update_voronoi(self.x_weights, self.voronoi_x)

        # Step 4: Update conscience for x
        self._update_conscience(bmu_x, self.x_winning_freq, self.x_conscience_bias)

        # Step 5.1: Find BMU for both conditional SOMs (needed for scoring in Step 8)
        bmu_x1 = self._find_bmu(observation, self.x1_weights, self.x1_conscience_bias)
        bmu_x0 = self._find_bmu(observation, self.x0_weights, self.x0_conscience_bias)

        # x1 uses self.lr; x0 uses its own slow base rate
        lr_x0 = self.lr_x0
        lr_x1 = self.lr_x1

        # Update x1 SOM only when instruction == 1.0
        if instruction == 1.0:
            # Step 5a.2: Update weights
            self._update_weights(bmu_x1, observation, self.x1_weights, lr=lr_x1)

            # Step 5a.3: Update voronoi_x1
            self._update_voronoi(self.x1_weights, self.voronoi_x1)

            # Step 5a.4: Update conscience for x1
            self._update_conscience(bmu_x1, self.x1_winning_freq, self.x1_conscience_bias)

        # Update x0 SOM only when instruction == 0.0
        if instruction == 0.0:
            # Step 5b.2: Update weights
            self._update_weights(bmu_x0, observation, self.x0_weights, lr=lr_x0)

            # Step 5b.3: Update voronoi_x0
            self._update_voronoi(self.x0_weights, self.voronoi_x0)

            # Step 5b.4: Update conscience for x0
            self._update_conscience(bmu_x0, self.x0_winning_freq, self.x0_conscience_bias)

        # Step 6: Calculate the inverse Voronoi cell size (1 / voronoi) for x, x1, and x0
        self.inv_voronoi_x = np.where(self.voronoi_x != 0, 1.0 / np.where(self.voronoi_x != 0, self.voronoi_x, 1.0), 0.0)
        self.inv_voronoi_x1 = np.where(self.voronoi_x1 != 0, 1.0 / np.where(self.voronoi_x1 != 0, self.voronoi_x1, 1.0), 0.0)
        self.inv_voronoi_x0 = np.where(self.voronoi_x0 != 0, 1.0 / np.where(self.voronoi_x0 != 0, self.voronoi_x0, 1.0), 0.0)

        # Step 7: Calculate the prior P(instruction = 1)
        diff_10 = self.inv_voronoi_x1 - self.inv_voronoi_x0
        numerator = np.sum((self.inv_voronoi_x - self.inv_voronoi_x0) * diff_10)
        denominator = np.sum(diff_10 ** 2)
        if denominator != 0:
            self.prior_instruction = float(np.clip(numerator / denominator, 0.0, 1.0))
        else:
            self.prior_instruction = 0.5

        # Step 8: Calculate X0 and X1 scores for this observation
        self.score_x1 = self.prior_instruction / self.voronoi_x1[bmu_x1] if self.voronoi_x1[bmu_x1] != 0 else 0.0
        self.score_x0 = (1 - self.prior_instruction) / self.voronoi_x0[bmu_x0] if self.voronoi_x0[bmu_x0] != 0 else 0.0

        if self.visualize:
            self._viz_step_count += 1
            if self._viz_step_count % self._viz_update_interval == 0:
                self._update_viz(instruction)

        # Step 9: Return predicted instruction
        return 1.0 if self.score_x1 > self.score_x0 else 0.0

    def _init_viz(self):
        self._ts_instruction = deque(maxlen=200)
        self._ts_score_x0 = deque(maxlen=200)
        self._ts_score_x1 = deque(maxlen=200)
        self._ts_prior = deque(maxlen=200)

        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        pg.setConfigOptions(antialias=True)
        self._win = pg.GraphicsLayoutWidget(title='SOM1D Dashboard')
        self._win.resize(1800, 900)

        x_idx = np.arange(self.resolution, dtype=float)
        bar_w = 0.7
        colors = [(180, 130, 70), (80, 127, 255), (60, 180, 75)]
        col_titles = ['x (Marginal)', 'x1 (instruction=1)', 'x0 (instruction=0)']
        ts_titles = ['Instruction', 'Score x1', 'Score x0']
        weights_init = [self.x_weights, self.x1_weights, self.x0_weights]
        freq_init = [self.x_winning_freq, self.x1_winning_freq, self.x0_winning_freq]

        self._weight_curves = []
        self._freq_bars = []
        self._ts_curves = []
        self._plots_freq = []
        self._plots_ts = []
        self._plots_weights = []

        for col in range(3):
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

            # Row 2: time series
            p = self._win.addPlot(row=2, col=col)
            p.setTitle(ts_titles[col])
            p.setLabel('bottom', 'Step')
            p.showGrid(y=True, alpha=0.3)
            p.enableAutoRange('y')
            c = p.plot([], pen=pg.mkPen(color=(r, g, b), width=1))
            self._ts_curves.append(c)
            self._plots_ts.append(p)

        self._win.show()

        # Row 3: prior P(instruction=1) — spans all 3 columns
        p = self._win.addPlot(row=3, col=0, colspan=3)
        p.setTitle('Prior P(instruction=1)')
        p.setLabel('bottom', 'Step')
        p.setLabel('left', 'Prior')
        p.showGrid(y=True, alpha=0.3)
        p.setYRange(0.0, 1.0)
        p.addLine(y=0.5, pen=pg.mkPen(color=(150, 150, 150), width=1, style=pg.QtCore.Qt.PenStyle.DashLine))
        self._prior_curve = p.plot([], pen=pg.mkPen(color=(220, 80, 220), width=2))
        self._plot_prior = p

        self._win.show()

    def _update_viz(self, instruction):
        self._ts_instruction.append(float(instruction))
        self._ts_score_x0.append(self.score_x0)
        self._ts_score_x1.append(self.score_x1)
        self._ts_prior.append(self.prior_instruction)

        x_idx = np.arange(self.resolution, dtype=float)
        weights = [self.x_weights, self.x1_weights, self.x0_weights]
        freqs = [self.x_winning_freq, self.x1_winning_freq, self.x0_winning_freq]
        ts_data = [self._ts_instruction, self._ts_score_x1, self._ts_score_x0]

        for col in range(3):
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

            ts = np.array(ts_data[col])
            self._ts_curves[col].setData(np.arange(len(ts), dtype=float), ts)
            if len(ts) > 0:
                cur_max_ts = float(np.max(ts))
                cur_min_ts = float(np.min(ts))
                pad = (cur_max_ts - cur_min_ts) * 0.1 or 0.1
                self._plots_ts[col].setYRange(cur_min_ts - pad, cur_max_ts + pad)

        prior_arr = np.array(self._ts_prior)
        self._prior_curve.setData(np.arange(len(prior_arr), dtype=float), prior_arr)

        self._app.processEvents()
