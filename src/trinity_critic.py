import sys
import os
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from collections import deque
sys.path.insert(0, os.path.dirname(__file__))
from som_1d import SOM1D


class TrinityCritic:
    def __init__(
        self,
        n_inputs,
        input_ranges=None,
        resolution=10,
        lr_x=0.001,
        lr_x1=0.001,
        neighborhood_decay=10,
        conscience_factor=0.5,
        conscience_lr=0.001,
        visualize=False,
        viz_update_interval=1,
    ):
        """
        Parameters
        ----------
        n_inputs : int
            Number of scalar input variables (observation dimensionality).
        input_ranges : list of (float, float)
            [(min, max), ...] — one pair per input variable.
        resolution : int
            Number of neurons per SOM1D.
        lr_x : float
            Learning rate for the marginal SOM (x).
        lr_x1 : float
            Learning rate for the conditional SOM (x1).
        neighborhood_decay : float
            Neighborhood decay constant for SOM1D weight updates.
        conscience_factor : float
            Conscience factor for the conscience mechanism.
        conscience_lr : float
            Learning rate for the conscience mechanism.
        """
        if input_ranges is not None and len(input_ranges) != n_inputs:
            raise ValueError(
                f"input_ranges length ({len(input_ranges)}) must match n_inputs ({n_inputs})"
            )
        self.visualize = visualize
        self._viz_update_interval = viz_update_interval
        self._viz_step_count = 0
        self.n_inputs = n_inputs
        self.soms = [
            SOM1D(
                resolution=resolution,
                lr_x=lr_x,
                lr_x1=lr_x1,
                neighborhood_decay=neighborhood_decay,
                conscience_factor=conscience_factor,
                conscience_lr=conscience_lr,
                input_min=input_ranges[i][0] if input_ranges is not None else None,
                input_max=input_ranges[i][1] if input_ranges is not None else None,
                visualize=self.visualize,
                viz_update_interval=self._viz_update_interval,
            )
            for i in range(n_inputs)
        ]
        if self.visualize:
            self._init_viz()

    def _aggregate_scores(self, observation, instruction):
        """
        Step all SOM1D members and collect per-SOM scores and specific mutual
        information values.

        Parameters
        ----------
        observation : array-like, shape (n_inputs,)
            The current observation vector.
        instruction : float
            The instruction signal (e.g. 1.0 or 0.0).

        Returns
        -------
        scores : list of float
            Probability-like score for instruction == 1 from each SOM1D.
        smi_values : list of float
            Specific mutual information value from each SOM1D.
        """
        scores = []
        smi_values = []
        per_predictions = []
        for i, som in enumerate(self.soms):
            posterior, score, smi = som.step(float(observation[i]), instruction)
            per_predictions.append(int(posterior))
            scores.append(score)
            smi_values.append(smi)
        return scores, smi_values, per_predictions

    def _calc_ensemble(self, scores, smi_values):
        if not scores:
            return 0, 0.0

        smi_sq = [s ** 2 for s in smi_values]
        total_smi_sq = sum(smi_sq)

        if total_smi_sq > 0.0:
            weights = [s / total_smi_sq for s in smi_sq]
        else:
            # Fall back to uniform weights when all SMI values are zero.
            n = len(scores)
            weights = [1.0 / n] * n

        ensemble_score = sum(w * s for w, s in zip(weights, scores))
        ensemble_prediction = 1 if ensemble_score >= 0.5 else 0
        return ensemble_prediction, ensemble_score

    def step(self, observation, instruction):
        """
        Step the critic for one timestep.

        Parameters
        ----------
        observation : array-like, shape (n_inputs,)
            The current observation vector.
        instruction : float
            The instruction signal (e.g. 1.0 or 0.0).

        Returns
        -------
        ensemble_prediction : int
            Binary ensemble prediction (0 or 1).
        ensemble_score : float
            Aggregated score: sum of log2(1 + score_i)^smi_i across all input SOMs,
            passed to ensemble_som for final prediction.
        scores : list of float
            Per-SOM probability-like scores.
        smi_values : list of float
            Per-SOM specific mutual information values.
        per_predictions : list of int
            Per-SOM binary posterior predictions (posterior_instruction from each SOM1D).
        """
        scores, smi_values, per_predictions = self._aggregate_scores(observation, instruction)
        ensemble_prediction, ensemble_score = self._calc_ensemble(scores, smi_values)

        if self.visualize:
            self._viz_step_count += 1
            if self._viz_step_count % self._viz_update_interval == 0:
                self._update_viz(ensemble_score, ensemble_prediction)

        return ensemble_prediction, ensemble_score, scores, smi_values, per_predictions

    def _init_viz(self):
        self._ts_ensemble_score = deque(maxlen=200)
        self._ts_ensemble_prediction = deque(maxlen=200)

        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        pg.setConfigOptions(antialias=True)
        self._win = pg.GraphicsLayoutWidget(title='TrinityCritic Dashboard')
        self._win.resize(900, 500)

        ts_specs = [
            ('Ensemble Score',      (255, 180,  50), False),
            ('Ensemble Prediction', ( 80, 127, 255), True),
        ]
        self._tc_ts_plots = []
        self._tc_ts_curves = []
        for i, (title, color, fixed_range) in enumerate(ts_specs):
            p = self._win.addPlot(row=i, col=0)
            p.setTitle(title)
            p.setLabel('bottom', 'Step')
            p.showGrid(y=True, alpha=0.3)
            if fixed_range:
                p.setYRange(-0.1, 1.1)
                p.addLine(y=0.5, pen=pg.mkPen(color=(150, 150, 150), width=1,
                          style=pg.QtCore.Qt.PenStyle.DashLine))
            else:
                p.enableAutoRange('y')
            c = p.plot([], pen=pg.mkPen(color=color, width=1))
            self._tc_ts_curves.append(c)
            self._tc_ts_plots.append(p)

        self._win.show()

    def _update_viz(self, ensemble_score, ensemble_prediction):
        self._ts_ensemble_score.append(float(ensemble_score))
        self._ts_ensemble_prediction.append(float(ensemble_prediction))

        ts_data = [self._ts_ensemble_score, self._ts_ensemble_prediction]
        for i, ts_deque in enumerate(ts_data):
            ts = np.array(ts_deque)
            self._tc_ts_curves[i].setData(np.arange(len(ts), dtype=float), ts)
            if i == 0 and len(ts) > 0:  # auto-range for ensemble score
                cur_max = float(np.max(ts))
                cur_min = float(np.min(ts))
                pad = (cur_max - cur_min) * 0.1 or 0.1
                self._tc_ts_plots[i].setYRange(cur_min - pad, cur_max + pad)

        self._app.processEvents()
