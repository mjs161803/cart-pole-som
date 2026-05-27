import sys
import os
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

    def step(self, observation, instruction):
        """
        Step all SOM1D members with the corresponding scalar from observation.

        Parameters
        ----------
        observation : array-like, shape (n_inputs,)
            The current observation vector.
        instruction : float
            The instruction signal (e.g. 1.0 or 0.0).

        Returns
        -------
        scores : list of float
            score_x1 returned by each SOM1D.
        posteriors : list of float
            posterior_instruction returned by each SOM1D.
        """
        scores = []
        posteriors = []
        for i, som in enumerate(self.soms):
            score, posterior = som.step(float(observation[i]), instruction)
            scores.append(score)
            posteriors.append(posterior)
        return scores, posteriors
