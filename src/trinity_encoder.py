import numpy as np
from collections import deque
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from trinity_critic import TrinityCritic
from entropy_neuron import EntropyNeuron


class TrinityEncoder():
    def __init__(self,
                 observation_dim=None,
                 learning_rate=0.01,
                 semantic_codelength=8,
                 encoder_resolution=10,
                 conscience_factor=0.5,
                 conscience_lr=0.01,
                 prior_ema_alpha=0.001,
                 encoder_visualize=False,
                 encoder_viz_update_interval=100,
                 feature_names=None,
                 ):
        self.observation_dim = observation_dim
        self.learning_rate = learning_rate
        self.semantic_codelength = semantic_codelength
        self.encoder_resolution = encoder_resolution
        self.conscience_factor = conscience_factor
        self.conscience_lr = conscience_lr
        self.prior_ema_alpha = prior_ema_alpha
        self.visualize = encoder_visualize
        self.viz_update_interval = encoder_viz_update_interval
        self.feature_names = feature_names

        self.layers = [
            (
                TrinityCritic(
                    n_inputs=observation_dim,
                    resolution=encoder_resolution,
                    lr=learning_rate,
                    conscience_factor=conscience_factor,
                    conscience_lr=conscience_lr,
                    prior_ema_alpha=prior_ema_alpha,
                    critic_visualize=True,
                    critic_viz_update_interval=encoder_viz_update_interval,
                    feature_names=feature_names,
                ),
                EntropyNeuron(
                    learning_rate=learning_rate,
                    conscience_learning_rate=conscience_lr,
                    conscience_factor=conscience_factor,
                ),
            )
            for _ in range(semantic_codelength - 1)
        ]
        self.prev_observation = None
        self.semantic_code = np.zeros(semantic_codelength)
        self.prev_semantic_code = np.zeros(semantic_codelength)

        if self.visualize:
            self._viz_step_count = 0
            self._waterfall_rows = 200
            self._waterfall_history = deque(maxlen=self._waterfall_rows)
            self._init_viz()

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _init_viz(self):
        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        pg.setConfigOptions(antialias=True)
        self._win = pg.GraphicsLayoutWidget(title='TrinityEncoder Dashboard')
        self._win.resize(900, 800)

        # --- Waterfall (top) ---
        self._wf_plot = self._win.addPlot(row=0, col=0)
        self._wf_plot.setTitle(
            f'Semantic Code Waterfall  '
            f'(x = bit index 0–{self.semantic_codelength - 1},  y = time,  newest at top)'
        )
        self._wf_plot.setLabel('left', 'Time step (newest ↑)')
        self._wf_plot.setLabel('bottom', 'Bit index')
        self._wf_plot.getAxis('bottom').setTicks(
            [[(i, str(i)) for i in range(self.semantic_codelength)]]
        )

        self._waterfall_img = pg.ImageItem()
        self._wf_plot.addItem(self._waterfall_img)
        self._wf_plot.setXRange(-0.5, self.semantic_codelength - 0.5, padding=0)
        self._wf_plot.setYRange(0, self._waterfall_rows, padding=0)

        # LUT: 0 → dark navy, 1 → bright yellow
        lut = np.zeros((256, 3), dtype=np.uint8)
        for k in range(256):
            t = k / 255.0
            lut[k] = [
                int((1 - t) * 15  + t * 255),
                int((1 - t) * 15  + t * 220),
                int((1 - t) * 60  + t *   0),
            ]
        self._waterfall_img.setLookupTable(lut)
        self._waterfall_img.setLevels([0, 1])

        self._win.show()

    def _track_state(self):
        """Record current code in waterfall history."""
        self._waterfall_history.append(self.semantic_code.copy())

    def _render_viz(self):
        """Redraw the waterfall. Called every viz_update_interval steps."""
        n = len(self._waterfall_history)
        if n > 0:
            arr = np.array(self._waterfall_history, dtype=np.float32)  # (n, codelength)
            if n < self._waterfall_rows:
                pad = np.zeros(
                    (self._waterfall_rows - n, self.semantic_codelength),
                    dtype=np.float32,
                )
                arr = np.vstack([pad, arr])
            # arr shape: (waterfall_rows, codelength)
            # imageAxisOrder='row-major' is set globally in cartpole_example.py,
            # so setImage(img) interprets img[row=y, col=x].
            # arr shape (waterfall_rows, codelength) → x=bit index, y=time ✓
            # y increases upward by default → arr[0] (oldest/pad) at bottom,
            # arr[-1] (newest) at top — classic waterfall convention
            self._waterfall_img.setImage(arr, autoLevels=False)

        self._app.processEvents()

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self, observation, z0):
        # The encoder receives the first semantic bit (z0) from the TrinityCritic and produces a semantic code based on the observation and z0.
        self.semantic_code[0] = z0

        for layer_idx, (trin_critic, entropy_neuron) in enumerate(self.layers): 
            semantic_code_idx = layer_idx + 1
            # Calculate prefix value of current semantic code and previous semantic code
            current_prefix = self.convert_code_to_integer(self.semantic_code[0:semantic_code_idx-1])
            prev_prefix = self.convert_code_to_integer(self.prev_semantic_code[0:semantic_code_idx-1])
            semantic_delta = current_prefix - prev_prefix
            positive_transition = 1 if semantic_delta == 1 else 0         

            # z_1 = (previous value of the current semantic bit) OR (prefix_value - prev_prefix_value == 1)
            z1 = self.prev_semantic_code[semantic_code_idx] + positive_transition
            
            # get prediction of semantic bit based on observation and z1
            sb_prediction = trin_critic.step(observation=observation, instruction=z1)

            # ensure maximum entropy of semantic bit
            self.semantic_code[semantic_code_idx] = entropy_neuron(sb_prediction)

        if self.visualize:
            self._track_state()
            self._viz_step_count += 1
            if self._viz_step_count % self.viz_update_interval == 0:
                self._render_viz()

        return self.semantic_code
    
    def convert_code_to_integer(self, code):
        integer_value = 0
        for bit in code:
            integer_value = (integer_value << 1) | int(bit)
        return integer_value