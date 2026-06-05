import sys
import os
import math
sys.path.insert(0, os.path.dirname(__file__))
from som_1d import SOM1D


class TrinityCritic:
    def __init__(
        self,
        n_inputs,
        input_ranges=None,
        resolution=20,
        lr_x=0.001,
        lr_x1=0.001,
        neighborhood_decay=3,
        conscience_factor=0.5,
        conscience_lr=0.001,
        ensemble_threshold=0.5,
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
        ensemble_threshold : float
            Probability threshold applied after log-linear pooling to produce
            the binary instruction prediction (default 0.5).
        """
        if input_ranges is not None and len(input_ranges) != n_inputs:
            raise ValueError(
                f"input_ranges length ({len(input_ranges)}) must match n_inputs ({n_inputs})"
            )

        self.n_inputs = n_inputs
        self.ensemble_threshold = ensemble_threshold
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
                visualize=False,
            )
            for i in range(n_inputs)
        ]

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
        for i, som in enumerate(self.soms):
            score, smi = som.step(float(observation[i]), instruction)
            scores.append(score)
            smi_values.append(smi)
        return scores, smi_values

    def _calc_ensemble(self, scores, smi_values):
        """
        Combine per-SOM predictions via log-linear pooling weighted by specific
        mutual information, then apply a threshold to produce a binary prediction.

        Log-linear pooling:
            logit(P_ensemble) = sum_i( w_i * logit(p_i) )
        where w_i = smi_values[i].

        Parameters
        ----------
        scores : list of float
            Per-SOM probability-like score for instruction == 1.
        smi_values : list of float
            Per-SOM specific mutual information used as pooling weights.

        Returns
        -------
        prediction : int
            1 if ensemble probability >= self.ensemble_threshold, else 0.
        ensemble_prob : float
            The pooled probability before thresholding.
        """
        _EPS = 1e-7
        total_weight = sum(abs(w) for w in smi_values)
        log_odds = 0.0
        for p, w in zip(scores, smi_values):
            p_clipped = max(_EPS, min(1.0 - _EPS, float(p)))
            log_odds += w * math.log(p_clipped / (1.0 - p_clipped))
        if total_weight > 0:
            ensemble_prob = 1.0 / (1.0 + math.exp(-log_odds))
        else:
            ensemble_prob = 0.5
        prediction = 1 if ensemble_prob >= self.ensemble_threshold else 0
        return prediction, ensemble_prob

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
        prediction : int
            Binary ensemble prediction (0 or 1).
        ensemble_prob : float
            Log-linear pooled probability before thresholding.
        scores : list of float
            Per-SOM probability-like scores.
        smi_values : list of float
            Per-SOM specific mutual information values.
        """
        scores, smi_values = self._aggregate_scores(observation, instruction)
        prediction, ensemble_prob = self._calc_ensemble(scores, smi_values)
        return prediction, ensemble_prob, scores, smi_values
