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
        ensemble_threshold=0.5,
        threshold_lr=0.01,
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
            Initial threshold applied to the SMI-weighted ensemble score to
            produce the binary instruction prediction (default 0.5).
            Adapted online by _update_threshold.
        threshold_lr : float
            Step size for threshold adaptation. On a false negative
            (instruction=1, prediction=0) the threshold moves down toward the
            ensemble score; on a false positive (instruction=0, prediction=1)
            it moves up toward the ensemble score (default 0.01).
        """
        if input_ranges is not None and len(input_ranges) != n_inputs:
            raise ValueError(
                f"input_ranges length ({len(input_ranges)}) must match n_inputs ({n_inputs})"
            )

        self.n_inputs = n_inputs
        self.ensemble_threshold = ensemble_threshold
        self.threshold_lr = threshold_lr
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
        Combine per-SOM scores via a weighted sum, where each weight is the
        corresponding SMI value normalised by the total SMI:

            ensemble_score = sum_i( (smi_i / sum_j(smi_j)) * score_i )

        Falls back to an unweighted average when all SMI values are zero.

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
            The weighted sum score before thresholding.
        """
        # Clip negative SMI values to 0.0 — only informative (positive-SMI) dimensions
        # contribute weight.  Negative SMI indicates the dimension is currently less
        # informative than the marginal and should not influence the ensemble.
        clipped_smi = [max(0.0, s) for s in smi_values]
        total_smi = sum(clipped_smi)
        if total_smi > 0:
            ensemble_score = sum((smi / total_smi) * float(score) for score, smi in zip(scores, clipped_smi))
        else:
            ensemble_score = sum(float(score) for score in scores) / len(scores) if scores else 0.0
        prediction = 1 if ensemble_score >= self.ensemble_threshold else 0
        return prediction, ensemble_score

    def _update_threshold(self, instruction, prediction, ensemble_score):
        """
        Adapt the ensemble threshold based on prediction errors.

        On a false negative (instruction=1, prediction=0) the threshold is
        shifted down toward ensemble_score.  On a false positive
        (instruction=0, prediction=1) it is shifted up toward ensemble_score.
        No adjustment is made on correct predictions.

        Parameters
        ----------
        instruction : float
            The true instruction signal (1.0 or 0.0).
        prediction : int
            The binary prediction produced by _calc_ensemble.
        ensemble_score : float
            The raw weighted score before thresholding.
        """
        if instruction == 1.0 and prediction == 0:
            # False negative: threshold is too high; move it down toward the score.
            self.ensemble_threshold += self.threshold_lr * (ensemble_score - self.ensemble_threshold)

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
        ensemble_score : float
            SMI-weighted sum of per-SOM scores, before thresholding.
        scores : list of float
            Per-SOM probability-like scores.
        smi_values : list of float
            Per-SOM specific mutual information values.
        """
        scores, smi_values = self._aggregate_scores(observation, instruction)
        prediction, ensemble_score = self._calc_ensemble(scores, smi_values)
        if instruction is not None:
            self._update_threshold(float(instruction), prediction, ensemble_score)
        return prediction, ensemble_score, scores, smi_values
