import os
import sys

# Force MuJoCo to use EGL for offscreen rendering; bypasses GLFW/Wayland (libdecor) entirely
os.environ["MUJOCO_GL"] = "egl"

os.environ["QT_QPA_PLATFORM"] = "xcb"          # force xcb; overrides any shell Wayland setting
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts.warning=false")

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
    resolution=80,
    lr_x=0.001,
    lr_x1=0.001,
    neighborhood_decay=10,
    conscience_factor=0.5,
    conscience_lr=0.001,
)

# Recall & precision tracking: per SOM1D
#   _tp_count        — instruction=1 AND posterior=1 (true positives)
#   _instruction1_count — instruction=1 (TP + FN, recall denominator)
#   _predicted1_count   — posterior=1  (TP + FP, precision denominator)
_tp_count = np.zeros(n_inputs, dtype=np.int64)
_instruction1_count = np.zeros(n_inputs, dtype=np.int64)
_predicted1_count = np.zeros(n_inputs, dtype=np.int64)


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

    # Access state and reward for your custom AI model
    observation = time_step.observation
    reward = time_step.reward

    # Extract pole angle from cos/sin components: 0 = vertical (up), ±pi = hanging down
    pole_angle = np.arctan2(observation['position'][2], observation['position'][1])
    instruction = 1.0 if abs(pole_angle) <= np.deg2rad(5.0) else 0.0

    # Flatten the full observation and step the TrinityCritic
    obs_flat = _process_obs(observation)
    scores, posteriors = critic.step(obs_flat, instruction)

    # Update recall & precision counters
    for i, posterior in enumerate(posteriors):
        if instruction == 1.0:
            _instruction1_count[i] += 1
        if posterior == 1.0:
            _predicted1_count[i] += 1
        if instruction == 1.0 and posterior == 1.0:
            _tp_count[i] += 1

    # Print per-SOM1D recall & precision every 50000 steps, then reset counters
    if _step % 50000 == 0:
        print(f"\n--- Step {_step} | SOM1D recall & precision (last 50k steps) ---")
        print(f"  {'#':>3}  {'Feature':<26s}  {'range':<16s}  {'recall':>7}  {'precision':>9}  tp/actual  tp/predicted")
        print(f"  {'-'*3}  {'-'*26}  {'-'*16}  {'-'*7}  {'-'*9}  {'-'*9}  {'-'*12}")
        for i in range(n_inputs):
            tp    = int(_tp_count[i])
            actual = int(_instruction1_count[i])
            pred   = int(_predicted1_count[i])
            recall    = 100.0 * tp / actual if actual > 0 else 0.0
            precision = 100.0 * tp / pred   if pred   > 0 else 0.0
            lo = critic.soms[i]._input_min
            hi = critic.soms[i]._input_max
            range_str = f"[{lo:.3g}, {hi:.3g}]" if lo is not None else "[not yet seen]"
            print(f"  [{i:2d}] {_input_labels[i]:<26s}  {range_str:<16s}  {recall:6.1f}%  {precision:8.1f}%  {tp}/{actual:<7}  {tp}/{pred}")
        _tp_count[:] = 0
        _instruction1_count[:] = 0
        _predicted1_count[:] = 0

    # 3. Visual Rendering: render cartpole camera to PyQtGraph window
    # if _step % _render_interval == 0:
    #     pixels = env.physics.render(camera_id=0, height=480, width=640)
    #     _cam_img.setImage(pixels)
    #     _app.processEvents()