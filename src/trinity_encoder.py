import numpy as np
import networkx as nx
from collections import deque
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from trinity_critic import TrinityCritic


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

        # Force visualize=False on layers — the encoder owns its own dashboard
        self.layers = [
            TrinityCritic(
                n_inputs=observation_dim,
                resolution=encoder_resolution,
                lr=learning_rate,
                conscience_factor=conscience_factor,
                conscience_lr=conscience_lr,
                prior_ema_alpha=prior_ema_alpha,
                critic_visualize=False,
                critic_viz_update_interval=encoder_viz_update_interval,
                feature_names=feature_names,
            )
            for _ in range(semantic_codelength)
        ]
        self.semantic_code = np.zeros(semantic_codelength)
        self.prev_semantic_code = np.zeros(semantic_codelength)

        if self.visualize:
            self._viz_step_count = 0
            self._waterfall_rows = 200
            self._waterfall_history = deque(maxlen=self._waterfall_rows)
            self._state_graph = nx.DiGraph()
            self._sm_layout_pos = {}
            self._prev_state_id = None
            self._sm_graph_dirty = False
            self._sm_node_labels = []
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

        # --- State machine (bottom) ---
        self._sm_plot = self._win.addPlot(row=1, col=0)
        self._sm_plot.setTitle(
            'State Machine  (node id = decimal value of semantic code,  '
            'current state = orange)'
        )
        self._sm_plot.hideAxis('left')
        self._sm_plot.hideAxis('bottom')

        self._sm_edge_item = pg.PlotDataItem(
            pen=pg.mkPen(color=(160, 160, 160), width=1)
        )
        self._sm_plot.addItem(self._sm_edge_item)

        self._sm_node_scatter = pg.ScatterPlotItem(
            size=18,
            pen=pg.mkPen(color=(255, 255, 255), width=1),
        )
        self._sm_plot.addItem(self._sm_node_scatter)

        self._win.show()

    def _semantic_code_to_int(self, code):
        """Convert binary code array to integer with bit 0 as MSB."""
        powers = 2 ** np.arange(self.semantic_codelength - 1, -1, -1)
        return int(np.dot(code, powers))

    def _track_state(self):
        """Record current code in waterfall history and update the state graph."""
        self._waterfall_history.append(self.semantic_code.copy())

        state_id = self._semantic_code_to_int(self.semantic_code)

        if state_id not in self._state_graph.nodes:
            self._state_graph.add_node(state_id)
            self._sm_graph_dirty = True

        if self._prev_state_id is not None:
            if not self._state_graph.has_edge(self._prev_state_id, state_id):
                self._state_graph.add_edge(self._prev_state_id, state_id, weight=1)
                self._sm_graph_dirty = True
            else:
                self._state_graph[self._prev_state_id][state_id]['weight'] += 1

        self._prev_state_id = state_id

    def _render_viz(self):
        """Redraw both panels. Called every viz_update_interval steps."""
        # --- Waterfall ---
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
            # ImageItem: setImage(img) where img[x, y] -> x=horizontal, y=vertical
            # arr.T: shape (codelength, waterfall_rows) -> x=bit index, y=time ✓
            # y increases upward by default -> arr[0] (oldest/pad) at bottom,
            # arr[-1] (newest) at top — classic waterfall convention
            self._waterfall_img.setImage(arr.T, autoLevels=False)

        # --- State machine ---
        nodes = list(self._state_graph.nodes())
        if not nodes:
            self._app.processEvents()
            return

        if self._sm_graph_dirty:
            fixed = [nd for nd in self._sm_layout_pos if nd in self._state_graph.nodes]
            new_pos = nx.spring_layout(
                self._state_graph,
                pos=self._sm_layout_pos if self._sm_layout_pos else None,
                fixed=fixed if fixed else None,
                seed=42,
                k=2.0,
            )
            self._sm_layout_pos.update(new_pos)
            self._sm_graph_dirty = False

        pos = self._sm_layout_pos

        # Edges as nan-separated line segments
        edge_x, edge_y = [], []
        for u, v in self._state_graph.edges():
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, np.nan]
                edge_y += [y0, y1, np.nan]
        if edge_x:
            self._sm_edge_item.setData(
                x=np.array(edge_x, dtype=float),
                y=np.array(edge_y, dtype=float),
            )

        # Nodes — current state in orange, others in blue
        curr = self._prev_state_id
        valid_nodes = [nd for nd in nodes if nd in pos]
        node_x = [pos[nd][0] for nd in valid_nodes]
        node_y = [pos[nd][1] for nd in valid_nodes]
        brushes = [
            pg.mkBrush(255, 120, 40, 230) if nd == curr
            else pg.mkBrush(60, 100, 220, 200)
            for nd in valid_nodes
        ]
        self._sm_node_scatter.setData(x=node_x, y=node_y, brush=brushes)
        self._sm_plot.enableAutoRange()

        # Node labels (decimal state id below each node)
        for lbl in self._sm_node_labels:
            self._sm_plot.removeItem(lbl)
        self._sm_node_labels.clear()

        for nd in valid_nodes:
            x, y = pos[nd]
            lbl = pg.TextItem(text=str(nd), anchor=(0.5, 1.6), color=(220, 220, 220))
            lbl.setPos(x, y)
            self._sm_plot.addItem(lbl)
            self._sm_node_labels.append(lbl)

        self._app.processEvents()

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self, observation, z0):
        # The encoder receives the first semantic bit (z0) from the TrinityCritic and produces a semantic code based on the observation and z0.
        self.semantic_code[0] = z0

        for i in range(self.semantic_codelength):
            if i == self.semantic_codelength - 1:
                break
            powers = 2 ** np.arange(i, -1, -1)  # length i+1, bit 0 is MSB
            code_val = int(np.dot(self.semantic_code[0:i+1], powers))
            prev_val = int(np.dot(self.prev_semantic_code[0:i+1], powers))
            enable = 1 if (code_val - prev_val) == 1 else 0
            next_semantic_bit, _, _, _, _ = self.layers[i].step(observation, enable)
            self.semantic_code[i+1] = next_semantic_bit

        self.prev_semantic_code = self.semantic_code.copy()

        if self.visualize:
            self._track_state()
            self._viz_step_count += 1
            if self._viz_step_count % self.viz_update_interval == 0:
                self._render_viz()

        return self.semantic_code