import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from collections import deque


class SOM1D:
    def __init__(
        self,
        resolution=10,
        initial_learning_rate=0.01,
        conscience_gamma=0.1,
        visualize=False,
        viz_update_interval=1,
    ):
        self.resolution = resolution
        self.initial_learning_rate = initial_learning_rate
        self.current_learning_rate = initial_learning_rate
        self.half_learning_rate = initial_learning_rate * 0.5
        self.conscience_gamma = conscience_gamma
        self.conscience_gamma_tn = conscience_gamma / 2.0
        
        # Weights: monotonically increasing, equally spaced
        self.x_weights = np.linspace(-np.pi, np.pi, resolution)
        self.x1_weights = np.linspace(-np.pi, np.pi, resolution)
        self.x0_weights = np.linspace(-np.pi, np.pi, resolution)

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
        self.prior_instruction = 0.0
        self.score_x1 = 0.0
        self.score_x0 = 0.0

        # Bias for conscience mechanism
        self.activation_bias_x = np.zeros(resolution)
        self.activation_bias_x1 = np.zeros(resolution)
        self.activation_bias_x0 = np.zeros(resolution)

        # Win count for visualization and conscience monitoring
        self.wins_x = np.zeros(resolution)
        self.wins_x1 = np.zeros(resolution)
        self.wins_x0 = np.zeros(resolution)

        self.visualize = visualize
        self._viz_update_interval = viz_update_interval
        self._viz_step_count = 0
        if self.visualize:
            self._init_viz()

    def _find_bmu(self, value, weights, activation_bias):
        distances = np.abs(weights - value) + activation_bias
        return np.argmin(distances)

    def _update_weights(self, bmu_idx, value, weights):
        if bmu_idx > 0:
            weights[bmu_idx - 1] += self.half_learning_rate * (value - weights[bmu_idx - 1])
        if bmu_idx < self.resolution - 1:
            weights[bmu_idx + 1] += self.half_learning_rate * (value - weights[bmu_idx + 1])

    def _update_voronoi(self, bmu_idx, weights, voronoi):
        neighbors = []
        if bmu_idx > 0:
            neighbors.append(bmu_idx - 1)
        if bmu_idx < self.resolution - 1:
            neighbors.append(bmu_idx + 1)
        dists = []
        for n_idx in neighbors:
            d = abs(weights[bmu_idx] - weights[n_idx])
            dists.append(d)
            voronoi[n_idx] = (voronoi[n_idx] + d) / 2
        if dists:
            voronoi[bmu_idx] = (voronoi[bmu_idx] + sum(dists) / len(dists)) / 2

    def _update_conscience(self, bmu_idx, wins, activation_bias, voronoi):
        wins[bmu_idx] += 1
        for i in range(self.resolution):
            if i == bmu_idx:
                activation_bias[i] += self.conscience_gamma * voronoi[i]
            else:
                activation_bias[i] -= self.conscience_gamma_tn * voronoi[i]

    def step(self, observation, instruction):
        # Step 1: Find BMU in x_weights
        bmu_x = self._find_bmu(observation, self.x_weights, self.activation_bias_x)

        # Step 2: Update BMU weight
        self.x_weights[bmu_x] += self.current_learning_rate * (observation - self.x_weights[bmu_x])

        # Step 3: Update neighborhood weights
        self._update_weights(bmu_x, observation, self.x_weights)

        # Step 4: Update voronoi_x for topological neighbors of BMU
        self._update_voronoi(bmu_x, self.x_weights, self.voronoi_x)

        # Update conscience for x
        self._update_conscience(bmu_x, self.wins_x, self.activation_bias_x, self.voronoi_x)

        # Step 5: Find BMU for both conditional SOMs (needed for scoring in Step 8)
        bmu_x1 = self._find_bmu(observation, self.x1_weights, self.activation_bias_x1)
        bmu_x0 = self._find_bmu(observation, self.x0_weights, self.activation_bias_x0)

        # Update x1 SOM only when instruction == 1.0
        if instruction == 1.0:
            # Step 5.2: Update BMU weight
            self.x1_weights[bmu_x1] += self.current_learning_rate * (observation - self.x1_weights[bmu_x1])

            # Step 5.3: Update neighborhood weights
            self._update_weights(bmu_x1, observation, self.x1_weights)

            # Step 5.4: Update voronoi_x1 for topological neighbors of BMU
            self._update_voronoi(bmu_x1, self.x1_weights, self.voronoi_x1)

            # Update conscience for x1
            self._update_conscience(bmu_x1, self.wins_x1, self.activation_bias_x1, self.voronoi_x1)

        # Update x0 SOM only when instruction == 0.0
        if instruction == 0.0:
            # Step 5b.2: Update BMU weight
            self.x0_weights[bmu_x0] += self.current_learning_rate * (observation - self.x0_weights[bmu_x0])

            # Step 5b.3: Update neighborhood weights
            self._update_weights(bmu_x0, observation, self.x0_weights)

            # Step 5b.4: Update voronoi_x0 for topological neighbors of BMU
            self._update_voronoi(bmu_x0, self.x0_weights, self.voronoi_x0)

            # Update conscience for x0
            self._update_conscience(bmu_x0, self.wins_x0, self.activation_bias_x0, self.voronoi_x0)

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

        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        pg.setConfigOptions(antialias=True)
        self._win = pg.GraphicsLayoutWidget(title='SOM1D Dashboard')
        self._win.resize(1800, 900)

        x_idx = np.arange(self.resolution, dtype=float)
        bar_w = 0.7
        colors = [(180, 130, 70), (80, 127, 255), (60, 180, 75)]
        col_titles = ['x (Marginal)', 'x1 (instruction=1)', 'x0 (instruction=0)']
        ts_titles = ['Instruction', 'Score x0', 'Score x1']
        weights_init = [self.x_weights, self.x1_weights, self.x0_weights]
        freq_init = [self.wins_x, self.wins_x1, self.wins_x0]
        voronoi_init = [self.inv_voronoi_x, self.inv_voronoi_x1, self.inv_voronoi_x0]

        self._weight_curves = []
        self._freq_bars = []
        self._voronoi_bars = []
        self._ts_curves = []
        self._plots_freq = []
        self._plots_voronoi = []
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

            # Row 1: total wins (bar chart)
            p = self._win.addPlot(row=1, col=col)
            p.setTitle(f'{col_titles[col]} — Total Wins')
            p.setLabel('bottom', 'Neuron')
            p.showGrid(y=True, alpha=0.3)
            p.enableAutoRange('y')
            bar = pg.BarGraphItem(x=x_idx, height=freq_init[col].copy(),
                                  width=bar_w, brush=brush, pen=pg.mkPen(None))
            p.addItem(bar)
            self._freq_bars.append(bar)
            self._plots_freq.append(p)

            # Row 2: inverse Voronoi (bar chart)
            p = self._win.addPlot(row=2, col=col)
            p.setTitle(f'{col_titles[col]} — Inv. Voronoi')
            p.setLabel('bottom', 'Neuron')
            p.showGrid(y=True, alpha=0.3)
            p.enableAutoRange('y')
            bar = pg.BarGraphItem(x=x_idx, height=voronoi_init[col].copy(),
                                  width=bar_w, brush=brush, pen=pg.mkPen(None))
            p.addItem(bar)
            self._voronoi_bars.append(bar)
            self._plots_voronoi.append(p)

            # Row 3: time series
            p = self._win.addPlot(row=3, col=col)
            p.setTitle(ts_titles[col])
            p.setLabel('bottom', 'Step')
            p.showGrid(y=True, alpha=0.3)
            p.enableAutoRange('y')
            c = p.plot([], pen=pg.mkPen(color=(r, g, b), width=1))
            self._ts_curves.append(c)
            self._plots_ts.append(p)

        self._win.show()

    def _update_viz(self, instruction):
        self._ts_instruction.append(float(instruction))
        self._ts_score_x0.append(self.score_x0)
        self._ts_score_x1.append(self.score_x1)

        x_idx = np.arange(self.resolution, dtype=float)
        weights = [self.x_weights, self.x1_weights, self.x0_weights]
        freqs = [self.wins_x, self.wins_x1, self.wins_x0]
        voronois = [self.inv_voronoi_x, self.inv_voronoi_x1, self.inv_voronoi_x0]
        ts_data = [self._ts_instruction, self._ts_score_x0, self._ts_score_x1]

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

            self._voronoi_bars[col].setOpts(height=voronois[col])
            cur_max_v = float(np.max(voronois[col]))
            cur_min_v = float(np.min(voronois[col]))
            pad = (cur_max_v - cur_min_v) * 0.1 or 0.1
            self._plots_voronoi[col].setYRange(cur_min_v - pad, cur_max_v + pad)

            ts = np.array(ts_data[col])
            self._ts_curves[col].setData(np.arange(len(ts), dtype=float), ts)
            if len(ts) > 0:
                cur_max_ts = float(np.max(ts))
                cur_min_ts = float(np.min(ts))
                pad = (cur_max_ts - cur_min_ts) * 0.1 or 0.1
                self._plots_ts[col].setYRange(cur_min_ts - pad, cur_max_ts + pad)

        self._app.processEvents()
