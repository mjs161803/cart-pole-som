import os
import sys

# Force MuJoCo to use EGL for offscreen rendering; bypasses GLFW/Wayland (libdecor) entirely
os.environ["MUJOCO_GL"] = "egl"

os.environ["QT_QPA_PLATFORM"] = "xcb"          # force xcb; overrides any shell Wayland setting
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts.warning=false")

from dm_control import suite
import numpy as np
import pyqtgraph as pg
pg.setConfigOptions(imageAxisOrder='row-major')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from trinity_critic import TrinityCritic

# Fallback range for observation dimensions reported as unbounded by the spec
_FALLBACK_RANGE = (-10.0, 10.0)

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

# Human-readable names for each (obs_key, element_index) in the cartpole observation
_OBS_LABEL_MAP = {
    ('position', 0): 'Cart Position',
    ('position', 1): 'Pole cos(\u03b8)',
    ('position', 2): 'Pole sin(\u03b8)',
    ('velocity', 0): 'Cart Velocity',
    ('velocity', 1): 'Pole Angular Velocity',
    ('velocity', 2): 'Pole Angular Vel. (sin)',
}

# Parse the observation spec into a flat list of (min, max) ranges and labels
obs_spec = env.observation_spec()
_obs_keys = list(obs_spec.keys())
input_ranges = []
_input_labels = []
for key in _obs_keys:
    spec = obs_spec[key]
    n = int(np.prod(spec.shape))
    has_bounds = hasattr(spec, 'minimum') and hasattr(spec, 'maximum')
    if has_bounds:
        mins = np.atleast_1d(spec.minimum)
        maxs = np.atleast_1d(spec.maximum)
    for j in range(n):
        if has_bounds:
            lo, hi = float(mins[j]), float(maxs[j])
            if np.isfinite(lo) and np.isfinite(hi):
                input_ranges.append((lo, hi))
            else:
                input_ranges.append(_FALLBACK_RANGE)
        else:
            input_ranges.append(_FALLBACK_RANGE)
        _input_labels.append(_OBS_LABEL_MAP.get((key, j), f"{key}[{j}]"))

n_inputs = len(input_ranges)

# Instantiate the TrinityCritic (one SOM1D per scalar observation dimension)
critic = TrinityCritic(
    n_inputs=n_inputs,
    input_ranges=input_ranges,
    resolution=20,
    lr_x=0.001,
    lr_x1=0.001,
    neighborhood_decay=3,
    conscience_factor=0.5,
    conscience_lr=0.001,
)

# Recall tracking: per SOM1D — among instruction=1 steps, count correct posteriors
_instruction1_count = np.zeros(n_inputs, dtype=np.int64)
_correct_count = np.zeros(n_inputs, dtype=np.int64)


def _flatten_obs(observation):
    """Concatenate all observation arrays into a single 1-D vector."""
    return np.concatenate([np.atleast_1d(observation[k]) for k in _obs_keys])

# Camera render window
_cam_win = pg.GraphicsLayoutWidget(title='Cart-Pole Camera')
_cam_win.resize(640, 480)
_cam_vb = _cam_win.addViewBox()
_cam_vb.setAspectLocked(True)
_cam_vb.invertY(True)
_cam_img = pg.ImageItem()
_cam_vb.addItem(_cam_img)
_cam_win.show()
_app = pg.QtWidgets.QApplication.instance()

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

    # Access state and reward for your custom AI model
    observation = time_step.observation
    reward = time_step.reward

    # Extract pole angle from cos/sin components: 0 = vertical (up), ±pi = hanging down
    pole_angle = np.arctan2(observation['position'][2], observation['position'][1])
    instruction = 1.0 if abs(pole_angle) <= np.deg2rad(5.0) else 0.0

    # Flatten the full observation and step the TrinityCritic
    obs_flat = _flatten_obs(observation)
    scores, posteriors = critic.step(obs_flat, instruction)

    # Update recall counters (instruction=1 steps only)
    if instruction == 1.0:
        _instruction1_count += 1
        for i, posterior in enumerate(posteriors):
            if posterior == 1.0:
                _correct_count[i] += 1

    # Print per-SOM1D recall every 50000 steps, then reset counters
    if _step % 50000 == 0:
        print(f"\n--- Step {_step} | SOM1D recall (instruction=1, last 50k steps) ---")
        print(f"  {'#':>3}  {'Observation Variable':<26s}  {'recall':>8}  correct/total")
        print(f"  {'-'*3}  {'-'*26}  {'-'*8}  {'-'*13}")
        for i in range(n_inputs):
            total = int(_instruction1_count[i])
            acc = 100.0 * _correct_count[i] / total if total > 0 else 0.0
            print(f"  [{i:2d}] {_input_labels[i]:<26s}  {acc:6.1f}%   ({int(_correct_count[i])}/{total})")
        _instruction1_count[:] = 0
        _correct_count[:] = 0

    # 3. Visual Rendering: render cartpole camera to PyQtGraph window
    if _step % _render_interval == 0:
        pixels = env.physics.render(camera_id=0, height=480, width=640)
        _cam_img.setImage(pixels)
        _app.processEvents()