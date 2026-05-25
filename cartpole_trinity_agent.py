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
from som_1d import SOM1D

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

# Instantiate the SOM1D for pole angle (creates the PyQtGraph dashboard)
som = SOM1D(visualize=True, resolution=10, lr_x1=0.05, lr_x=0.001, lr_x0=0.001, neighborhood_decay=3, viz_update_interval=10)

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
_render_interval = 100  # render camera every N steps (higher = faster simulation)

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
    instruction = 1.0 if abs(pole_angle) <= np.deg2rad(45.0) else 0.0
    som.step(pole_angle, instruction)

    # 3. Visual Rendering: render cartpole camera to PyQtGraph window
    if _step % _render_interval == 0:
        pixels = env.physics.render(camera_id=0, height=480, width=640)
        _cam_img.setImage(pixels)
        _app.processEvents()