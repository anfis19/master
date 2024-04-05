import mujoco
import mujoco.viewer

import time
import itertools
import numpy as np
from typing import Callable, NamedTuple, Optional, Union, List

import mediapy as media
import matplotlib.pyplot as plt

from pathlib import Path

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

model_dir = Path("/home/anton/master/mujoco_menagerie/universal_robots_ur5e")
model_xml = model_dir / "scene.xml"


xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_path(str(model_xml))
data = mujoco.MjData(model)



m = model
d = data

path = np.arange(-1, -2, -0.001)
i = 0
end = len(path)


with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running(): #and time.time() - start < 30:
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    # with viewer.lock():
    #   viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
    
    if i < end:
      pos = path[i]
      print(path[i]) 
      d.qpos[1] = pos
      d.actuator("shoulder_pan").ctrl = 0.2
    i = i + 1

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()


    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)

# print('test')
# renderer = mujoco.Renderer(model)
# mujoco.mj_forward(model, data)
# renderer.update_scene(data)
# media.show_image(renderer.render())
