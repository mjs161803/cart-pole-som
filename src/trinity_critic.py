import math
import sys
import os
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
        per_predictions = []
        for i, som in enumerate(self.soms):
            posterior, score, smi = som.step(float(observation[i]), instruction)
            per_predictions.append(int(posterior))
            scores.append(score)
            smi_values.append(smi)
        return scores, smi_values, per_predictions

    def _calc_ensemble(self, scores, smi_values):
        """
        Combine per-SOM scores via a log-linear pool (logarithmic opinion pool),
        where each weight is the corresponding SMI value normalised by the total
        SMI:

            p_pool = prod(s_i ^ w_i) / (prod(s_i ^ w_i) + prod((1-s_i) ^ w_i))

        Computed in log-space for numerical stability.  Falls back to equal
        weights when all SMI values are zero.  Negative SMI values are clipped
        to 0.0 so non-informative dimensions contribute no weight.

        Parameters
        ----------
        scores : list of float
            Per-SOM probability-like score for instruction == 1.
        smi_values : list of float
            Per-SOM specific mutual information used as pooling weights.

        Returns
        -------
        prediction : int
            1 if ensemble score >= self.ensemble_threshold, else 0.
        ensemble_score : float
            The log-linear pooled score before thresholding.
        """
        if not scores:
            return 0, 0.0

        # Clip negative SMI values to 0.0 — only informative (positive-SMI) dimensions
        # contribute weight.  Negative SMI indicates the dimension is currently less
        # informative than the marginal and should not influence the ensemble.
        clipped_smi = [max(0.0, s) for s in smi_values]
        total_smi = sum(clipped_smi)
        if total_smi > 0:
            weights = [s / total_smi for s in clipped_smi]
        else:
            weights = [1.0 / len(scores)] * len(scores)

        # Log-linear pool computed in log-space.
        # log_num = sum_i( w_i * log(s_i) )
        # log_den = sum_i( w_i * log(1 - s_i) )
        # ensemble_score = 1 / (1 + exp(log_den - log_num))
        _eps = 1e-12
        log_num = 0.0
        log_den = 0.0
        for w, score in zip(weights, scores):
            s = max(_eps, min(1.0 - _eps, float(score)))
            log_num += w * math.log(s)
            log_den += w * math.log(1.0 - s)

        ensemble_score = 1.0 / (1.0 + math.exp(log_den - log_num))
        ensemble_prediction = 1 if ensemble_score > 0.5 else 0
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
            SMI-weighted log-linear pooled score, before thresholding.
        scores : list of float
            Per-SOM probability-like scores.
        smi_values : list of float
            Per-SOM specific mutual information values.
        per_predictions : list of int
            Per-SOM binary posterior predictions (posterior_instruction from each SOM1D).
        """
        scores, smi_values, per_predictions = self._aggregate_scores(observation, instruction)
        ensemble_prediction, ensemble_score = self._calc_ensemble(scores, smi_values)
        return ensemble_prediction, ensemble_score, scores, smi_values, per_predictions
