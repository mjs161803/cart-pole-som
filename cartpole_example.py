import os
import sys

# Configure rendering backend based on OS
if sys.platform == "linux":
    # EGL for offscreen rendering; bypasses GLFW/Wayland (libdecor) entirely
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["QT_QPA_PLATFORM"] = "xcb"          # force xcb; overrides any shell Wayland setting
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts.warning=false")
elif sys.platform == "darwin":
    # macOS: disable MuJoCo GL backend (valid dm_control value) since this script doesn't render MuJoCo frames
    os.environ["MUJOCO_GL"] = "disable"
elif sys.platform == "win32":
    # Windows: GLFW is the standard rendering backend for MuJoCo
    os.environ["MUJOCO_GL"] = "glfw"

from dm_control import suite
import numpy as np
import pyqtgraph as pg
pg.setConfigOptions(imageAxisOrder='row-major')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from trinity_agent import TrinityAgent

# ---------------------------------------------------------------------------
# Rendering toggle — set to False for faster simulation
# ---------------------------------------------------------------------------
RENDER_CARTPOLE = False
RENDER_INTERVAL = 1   # render camera every N steps

# 1. Load the environment with an infinite time limit for online training
env = suite.load(
    domain_name="cartpole",
    task_name="swingup",
    task_kwargs={'time_limit': float('inf')}
)

# Retrieve the continuous action space specifications
action_spec = env.action_spec()

# Reset the environment to start
time_step = env.reset()

# ---------------------------------------------------------------------------
# Feature selection — comment out any line to exclude that feature from the
# critic.  _input_labels and _process_obs are derived automatically.
# ---------------------------------------------------------------------------
_feature_spec = [
    # (label,                   extractor)
    #('Cart Position',           lambda obs: float(obs['position'][0])),
    ('Pole Angle theta',        lambda obs: float(np.arctan2(obs['position'][2], obs['position'][1]))),
    #('Cart Velocity',           lambda obs: float(obs['velocity'][0])),
    #('Pole Angular Velocity',   lambda obs: float(obs['velocity'][1])),
]

_input_labels = [name for name, _ in _feature_spec]
n_inputs = len(_input_labels)


def _process_obs(observation):
    """Build the critic feature vector from the active _feature_spec entries."""
    return np.array([fn(observation) for _, fn in _feature_spec])


# Instantiate TrinityAgent
agent = TrinityAgent(
    observation_dim=n_inputs,
    critic_resolution=90,
    critic_lr=0.005,
    critic_conscience_factor=0.5,
    critic_conscience_lr=0.005,
    critic_prior_ema_alpha=0.001,
    critic_visualize=False,
    critic_viz_update_interval=1,
    feature_names=_input_labels,
    encoder_learning_rate=0.01,
    encoder_semantic_codelength=8,
    encoder_resolution=90,
    encoder_conscience_factor=0.5,
    encoder_conscience_lr=0.01,
    encoder_prior_ema_alpha=0.001,
    encoder_visualize=True,
    encoder_viz_update_interval=1,
)

# Ensemble-level prediction counters
_tp_count = 0
_instruction1_count = 0
_predicted1_count = 0

# Per-variable prediction counters
_per_tp         = np.zeros(n_inputs, dtype=int)
_per_actual1    = np.zeros(n_inputs, dtype=int)
_per_predicted1 = np.zeros(n_inputs, dtype=int)

# Per-SOM diagnostic accumulators (for average score/SMI display)
_score_sum = np.zeros(n_inputs)
_pmi_sum   = np.zeros(n_inputs)
_ensemble_score_sum  = 0.0
_diag_steps = 0

# Camera render window
if RENDER_CARTPOLE:
    _cam_win = pg.GraphicsLayoutWidget(title='Cart-Pole Camera')
    _cam_win.resize(640, 480)
    _cam_vb = _cam_win.addViewBox()
    _cam_vb.setAspectLocked(True)
    _cam_vb.invertY(True)
    _cam_img = pg.ImageItem()
    _cam_vb.addItem(_cam_img)
    _cam_win.show()
    _app = pg.QtWidgets.QApplication.instance()

# Semantic code bit accumulators
_semantic_code_sum = np.zeros(agent.encoder.semantic_codelength)

# Sinusoidal action parameters
_step = 0
_action_freq = .01   # cycles per step (adjust to taste)

# 2. Infinite training loop
while True:
    # Sinusoidal action in [-1, 1]
    action = np.array([np.sin(2 * np.pi * _action_freq * _step)])
    _step += 1

    # Step the environment forward
    time_step = env.step(action)

    # Access state 
    observation = time_step.observation
    # Flatten the observation dict into a feature vector for the critic, converting cos/sin to θ
    obs_flat = _process_obs(observation)

    # Define instruction based on pole angle: 1.0 if |θ| <= 5 degrees, else 0.0
    pole_angle = float(np.arctan2(observation['position'][2], observation['position'][1]))
    instruction = 1.0 if (abs(pole_angle) <= np.deg2rad(5.0) ) else 0.0
    
    # Step the TrinityAgent with the current observation and instruction
    semantic_code, ensemble_prediction, ensemble_score, scores, pmi_values, per_predictions = agent.step(obs_flat, instruction)

    # Update ensemble-level counters
    if instruction == 1.0:
        _instruction1_count += 1
    if ensemble_prediction == 1:
        _predicted1_count += 1
    if instruction == 1.0 and ensemble_prediction == 1:
        _tp_count += 1

    # Update per-variable counters
    for i in range(n_inputs):
        if instruction == 1.0:
            _per_actual1[i] += 1
        if per_predictions[i] == 1:
            _per_predicted1[i] += 1
        if instruction == 1.0 and per_predictions[i] == 1:
            _per_tp[i] += 1

    # Accumulate per-SOM diagnostics
    for i in range(n_inputs):
        _score_sum[i] += scores[i]
        _pmi_sum[i]   += pmi_values[i]
    _ensemble_score_sum  += ensemble_score
    _semantic_code_sum   += semantic_code
    _diag_steps += 1

    # Print diagnostics every 50000 steps, then reset counters
    if _step % 50000 == 0:
        recall    = 100.0 * _tp_count / _instruction1_count if _instruction1_count > 0 else 0.0
        precision = 100.0 * _tp_count / _predicted1_count   if _predicted1_count   > 0 else 0.0
        print(f"\n--- Step {_step} | Ensemble recall & precision (last 50k steps) ---")
        print(f"  Recall: {recall:.1f}%  |  Precision: {precision:.1f}%  |  "
              f"tp/actual: {_tp_count}/{_instruction1_count}  |  "
              f"tp/predicted: {_tp_count}/{_predicted1_count}")
        n = max(_diag_steps, 1)
        print(f"\n  {'#':>3}  {'Feature':<26s}  {'range':<18s}  {'avg score':>10}  {'avg PMI':>9}  {'recall':>7}  {'precision':>9}")
        print(f"  {'-'*3}  {'-'*26}  {'-'*18}  {'-'*10}  {'-'*9}  {'-'*7}  {'-'*9}")
        for i in range(n_inputs):
            lo = agent.critic.soms[i].obs_min
            hi = agent.critic.soms[i].obs_max
            range_str = f"[{lo:.3g}, {hi:.3g}]" if lo is not None else "[not yet seen]"
            avg_score = _score_sum[i] / n
            avg_pmi   = _pmi_sum[i]   / n
            per_recall    = 100.0 * _per_tp[i] / _per_actual1[i]    if _per_actual1[i]    > 0 else 0.0
            per_precision = 100.0 * _per_tp[i] / _per_predicted1[i] if _per_predicted1[i] > 0 else 0.0
            print(f"  [{i:2d}] {_input_labels[i]:<26s}  {range_str:<18s}  {avg_score:10.4f}  {avg_pmi:9.4f}  {per_recall:6.1f}%  {per_precision:8.1f}%")
        avg_ens_score = _ensemble_score_sum / n
        print(f"\n  Ensemble — avg score: {avg_ens_score:.4f}")
        avg_bits = _semantic_code_sum / n
        print(f"\n  Semantic code avg bits: [{', '.join(f'{v:.3f}' for v in avg_bits)}]")
        _tp_count = 0
        _instruction1_count = 0
        _predicted1_count = 0
        _per_tp[:]         = 0
        _per_actual1[:]    = 0
        _per_predicted1[:] = 0
        _score_sum[:] = 0.0
        _pmi_sum[:]   = 0.0
        _ensemble_score_sum  = 0.0
        _semantic_code_sum[:] = 0.0
        _diag_steps = 0

    # 3. Visual Rendering: render cartpole camera to PyQtGraph window
    if RENDER_CARTPOLE and _step % RENDER_INTERVAL == 0:
        pixels = env.physics.render(camera_id=0, height=480, width=640)
        _cam_img.setImage(pixels)
        _app.processEvents()