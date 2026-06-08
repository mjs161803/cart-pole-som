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
        self.ensemble_som = SOM1D(
            resolution=2,
            lr_x=0.01,
            lr_x1=0.01,
            neighborhood_decay=2,
            conscience_factor=0.5,
            conscience_lr=0.01,
            input_min=0,
            input_max=None,
            visualize=False,
        )

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

    def _calc_ensemble(self, scores, smi_values, instruction):
        if not scores:
            return 0, 0.0

        ensemble_score = 0.0
        for score, smi_value in zip(scores, smi_values):
            # Clip score to [0, inf) so the log2 base is always >= 1 (log2 >= 0),
            # keeping the result real for any real smi_value exponent.
            # Skip the term when the base is 0 and smi_value is negative (would be inf).
            base = math.log2(1.0 + max(0.0, score))
            if base > 0.0 or smi_value >= 0.0:
                ensemble_score += base ** smi_value

        ensemble_som_prediction, ensemble_som_score, ensemble_som_smi = self.ensemble_som.step(ensemble_score, instruction)
        return ensemble_som_prediction, ensemble_score

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
        ensemble_prediction, ensemble_score = self._calc_ensemble(scores, smi_values, instruction)
        return ensemble_prediction, ensemble_score, scores, smi_values, per_predictions
