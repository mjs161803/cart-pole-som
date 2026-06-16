import sys
import os
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from collections import deque
sys.path.insert(0, os.path.dirname(__file__))
from bayesian_neuron import BayesianNeuron


class TrinityCritic:
    def __init__(
        self,
        n_inputs,
        input_ranges=None,
        resolution=10,
        lr=0.001,
        conscience_factor=0.5,
        conscience_lr=0.001,
        prior_ema_alpha=0.001,
        ensemble_method='linear',
        critic_visualize=False,
        critic_viz_update_interval=1,
        feature_names=None,
    ):
        """
        Parameters
        ----------
        n_inputs : int
            Number of scalar input variables (observation dimensionality).
        input_ranges : list of (float, float)
            [(min, max), ...] — one pair per input variable.
        resolution : int
            Number of neurons per BayesianNeuron.
        lr : float
            Learning rate for weight updates.
        conscience_factor : float
            Conscience factor for the conscience mechanism.
        conscience_lr : float
            Learning rate for the conscience mechanism.
        prior_ema_alpha : float
            EMA decay rate for the prior P(instruction=1).
        """
        if input_ranges is not None and len(input_ranges) != n_inputs:
            raise ValueError(
                f"input_ranges length ({len(input_ranges)}) must match n_inputs ({n_inputs})"
            )
        if ensemble_method not in ('linear', 'loglinear'):
            raise ValueError(f"ensemble_method must be 'linear' or 'loglinear', got '{ensemble_method}'")
        self.ensemble_method = ensemble_method
        self.visualize = critic_visualize
        self._viz_update_interval = critic_viz_update_interval
        self._viz_step_count = 0
        self.n_inputs = n_inputs
        self.soms = [
            BayesianNeuron(
                resolution=resolution,
                lr=lr,
                conscience_factor=conscience_factor,
                conscience_lr=conscience_lr,
                prior_ema_alpha=prior_ema_alpha,
                obs_min=input_ranges[i][0] if input_ranges is not None else None,
                obs_max=input_ranges[i][1] if input_ranges is not None else None,
                visualize=self.visualize,
                viz_update_interval=self._viz_update_interval,
                feature_name=feature_names[i] if feature_names is not None else '',
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
            Probability-like score for instruction == 1 from each BayesianNeuron.
        pmi_values : list of float
            Pointwise mutual information value from each BayesianNeuron.
        """
        scores = []
        pmi_values = []
        per_predictions = []
        for i, som in enumerate(self.soms):
            prediction, posterior, pmi = som.step(float(observation[i]), instruction)
            per_predictions.append(prediction)
            scores.append(posterior)
            pmi_values.append(pmi)
        return scores, pmi_values, per_predictions

    def _calc_ensemble(self, scores, pmi_values):
        if self.ensemble_method == 'loglinear':
            return self._calc_ensemble_loglinear(scores, pmi_values)
        return self._calc_ensemble_linear(scores, pmi_values)

    def _calc_ensemble_linear(self, scores, pmi_values):
        if not scores:
            return 0, 0.0
        
        relevance = [2 ** s for s in pmi_values]
        total_relevance = sum(relevance)

        if total_relevance > 0.0:
            weights = [r / total_relevance for r in relevance]
        else:
            # Fall back to uniform weights when all PMI values are zero.
            n = len(scores)
            weights = [1.0 / n] * n

        ensemble_score = sum(w * s for w, s in zip(weights, scores))
        if not np.isfinite(ensemble_score):
            ensemble_score = 0.0
        ensemble_prediction = 1 if ensemble_score >= 0.5 else 0
        return ensemble_prediction, ensemble_score

    def _calc_ensemble_loglinear(self, scores, pmi_values):
        if not scores:
            return 0, 0.0

        safe_scores = [float(s) if np.isfinite(s) else 0.0 for s in scores]
        # Use raw PMI values as exponent weights; clip negatives and non-finite to 0.
        safe_pmi = [float(p) if np.isfinite(p) and p >= 0.0 else 0.0 for p in pmi_values]
        total_pmi = sum(safe_pmi)
        if total_pmi > 0.0:
            weights = [p / total_pmi for p in safe_pmi]
        else:
            n = len(scores)
            weights = [1.0 / n] * n

        # Formal log-linear (opinion-pool) definition:
        #   ensemble_score = (1/Z) * prod_i( s_i ^ w_i )
        # where Z normalises over both classes so the result is a valid probability:
        #   Z = prod_i(s_i^w_i) + prod_i((1-s_i)^w_i)
        eps = 1e-12
        unnorm_1 = float(np.exp(sum(w * np.log(max(s,       eps)) for w, s in zip(weights, safe_scores))))
        unnorm_0 = float(np.exp(sum(w * np.log(max(1.0 - s, eps)) for w, s in zip(weights, safe_scores))))
        z = unnorm_1 + unnorm_0
        ensemble_score = (unnorm_1 / z) if z > 0.0 else 0.0
        if not np.isfinite(ensemble_score):
            ensemble_score = 0.0
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
        pmi_values : list of float
            Per-SOM pointwise mutual information values.
        per_predictions : list of int
            Per-SOM binary posterior predictions (posterior_instruction from each BayesianNeuron).
        """
        scores, pmi_values, per_predictions = self._aggregate_scores(observation, instruction)
        ensemble_prediction, ensemble_score = self._calc_ensemble(scores, pmi_values)

        if self.visualize:
            self._viz_step_count += 1
            if self._viz_step_count % self._viz_update_interval == 0:
                self._update_viz(ensemble_score, ensemble_prediction)

        return ensemble_prediction, ensemble_score, scores, pmi_values, per_predictions

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
        self._ts_ensemble_score.append(float(ensemble_score) if np.isfinite(ensemble_score) else 0.0)
        self._ts_ensemble_prediction.append(float(ensemble_prediction))

        ts_data = [self._ts_ensemble_score, self._ts_ensemble_prediction]
        for i, ts_deque in enumerate(ts_data):
            ts = np.nan_to_num(np.array(ts_deque), nan=0.0, posinf=0.0, neginf=0.0)
            self._tc_ts_curves[i].setData(np.arange(len(ts), dtype=float), ts)
            if i == 0 and len(ts) > 0:  # auto-range for ensemble score
                cur_max = float(np.max(ts))
                cur_min = float(np.min(ts))
                pad = (cur_max - cur_min) * 0.1 or 0.1
                self._tc_ts_plots[i].setYRange(cur_min - pad, cur_max + pad)

        self._app.processEvents()
