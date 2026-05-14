import os
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts.warning=false")

from dm_control import suite
import numpy as np
import cv2 # For visualizing the render

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

# 2. Infinite training loop
while True:
    # Sample a random continuous action within the allowed bounds
    action = np.random.uniform(
        action_spec.minimum, 
        action_spec.maximum, 
        size=action_spec.shape
    )
    
    # Step the environment forward
    time_step = env.step(action)
    
    # Access state and reward for your custom AI model
    observation = time_step.observation
    reward = time_step.reward
    
    # 3. Visual Rendering: Extract RGB pixels and display them
    # camera_id=0 usually tracks the cart-pole dynamically
    pixels = env.physics.render(camera_id=0, height=480, width=640)
    
    # Convert RGB (dm_control) to BGR (OpenCV) for rendering in a window
    cv2.imshow("Cart-Pole Continuous Swingup", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()