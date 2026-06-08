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
    # macOS: EGL unavailable; GLFW uses native Cocoa, Qt uses Cocoa by default
    os.environ["MUJOCO_GL"] = "glfw"
elif sys.platform == "win32":
    # Windows: GLFW is the standard rendering backend for MuJoCo
    os.environ["MUJOCO_GL"] = "glfw"

from dm_control import suite
import numpy as np
# import pyqtgraph as pg
# pg.setConfigOptions(imageAxisOrder='row-major')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from trinity_critic import TrinityCritic

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

# Cartpole-specific feature vector: replace cos(θ)/sin(θ) with θ = arctan2(sin, cos)
# Features: [cart_pos, θ, cart_vel, pole_angular_vel]
# Input ranges are tracked dynamically by each SOM1D from observed data.
_input_labels = [
    'Cart Position',
    'Pole Angle θ',
    'Cart Velocity',
    'Pole Angular Velocity',
]
n_inputs = len(_input_labels)

# Instantiate the TrinityCritic (one SOM1D per scalar observation dimension)
critic = TrinityCritic(
    n_inputs=n_inputs,
    resolution=180,
    lr_x=0.001,
    lr_x1=0.001,
    neighborhood_decay=180,
    conscience_factor=0.5,
    conscience_lr=0.001,
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
_smi_sum   = np.zeros(n_inputs)
_ensemble_score_sum  = 0.0
_ensemble_som_smi_sum = 0.0
_diag_steps = 0


def _process_obs(observation):
    """Convert observation dict to feature vector, replacing cos/sin with θ."""
    cart_pos = float(observation['position'][0])
    theta    = float(np.arctan2(observation['position'][2], observation['position'][1]))
    cart_vel = float(observation['velocity'][0])
    pole_vel = float(observation['velocity'][1])
    return np.array([cart_pos, theta, cart_vel, pole_vel])

# Camera render window (commented out for faster simulation)
# _cam_win = pg.GraphicsLayoutWidget(title='Cart-Pole Camera')
# _cam_win.resize(640, 480)
# _cam_vb = _cam_win.addViewBox()
# _cam_vb.setAspectLocked(True)
# _cam_vb.invertY(True)
# _cam_img = pg.ImageItem()
# _cam_vb.addItem(_cam_img)
# _cam_win.show()
# _app = pg.QtWidgets.QApplication.instance()

# Sinusoidal action parameters
_step = 0
_action_freq = .01   # cycles per step (adjust to taste)
_render_interval = 1000  # render camera every N steps (higher = faster simulation)

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

    # Define instruction based on pole angle: 1.0 if |θ| ≤ 5 degrees, else 0.0
    pole_angle = obs_flat[1]
    instruction = 1.0 if abs(pole_angle) <= np.deg2rad(5.0) else 0.0

    # Step the TrinityCritic
    ensemble_prediction, ensemble_score, scores, smi_values, per_predictions = critic.step(obs_flat, instruction)

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
        _smi_sum[i]   += smi_values[i]
    _ensemble_score_sum  += ensemble_score
    _ensemble_som_smi_sum += critic.ensemble_som.smi
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
        print(f"\n  {'#':>3}  {'Feature':<26s}  {'range':<18s}  {'avg score':>10}  {'avg SMI':>9}  {'recall':>7}  {'precision':>9}")
        print(f"  {'-'*3}  {'-'*26}  {'-'*18}  {'-'*10}  {'-'*9}  {'-'*7}  {'-'*9}")
        for i in range(n_inputs):
            lo = critic.soms[i]._input_min
            hi = critic.soms[i]._input_max
            range_str = f"[{lo:.3g}, {hi:.3g}]" if lo is not None else "[not yet seen]"
            avg_score = _score_sum[i] / n
            avg_smi   = _smi_sum[i]   / n
            per_recall    = 100.0 * _per_tp[i] / _per_actual1[i]    if _per_actual1[i]    > 0 else 0.0
            per_precision = 100.0 * _per_tp[i] / _per_predicted1[i] if _per_predicted1[i] > 0 else 0.0
            print(f"  [{i:2d}] {_input_labels[i]:<26s}  {range_str:<18s}  {avg_score:10.4f}  {avg_smi:9.4f}  {per_recall:6.1f}%  {per_precision:8.1f}%")
        ens_lo = critic.ensemble_som._input_min
        ens_hi = critic.ensemble_som._input_max
        ens_range_str = f"[{ens_lo:.3g}, {ens_hi:.3g}]" if ens_lo is not None else "[not yet seen]"
        avg_ens_score = _ensemble_score_sum / n
        avg_ens_smi   = _ensemble_som_smi_sum / n
        print(f"\n  Ensemble SOM — input range: {ens_range_str}  |  avg raw score: {avg_ens_score:.4f}  |  avg SMI: {avg_ens_smi:.4f}")
        _tp_count = 0
        _instruction1_count = 0
        _predicted1_count = 0
        _per_tp[:]         = 0
        _per_actual1[:]    = 0
        _per_predicted1[:] = 0
        _score_sum[:] = 0.0
        _smi_sum[:]   = 0.0
        _ensemble_score_sum  = 0.0
        _ensemble_som_smi_sum = 0.0
        _diag_steps = 0

    # 3. Visual Rendering: render cartpole camera to PyQtGraph window
    # if _step % _render_interval == 0:
    #     pixels = env.physics.render(camera_id=0, height=480, width=640)
    #     _cam_img.setImage(pixels)
    #     _app.processEvents()