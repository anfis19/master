import mujoco as mj
import numpy as np
import mujoco.viewer as mj_viewer
from mujoco.glfw import glfw
from pathlib import Path
import time

import threading


from mujoco_base import MuJoCoBase

class Ur5e(MuJoCoBase):
    def __init__(self, xml_path):
        super().__init__(xml_path)

    def reset(self):
        # Set initial position of ball
        self.data.qpos[2] = 0.1

        # Set initial velocity of ball
        self.data.qvel[0] = 2.0
        self.data.qvel[2] = 5.0

        # Set camera configuration
        self.cam.azimuth = 90.0
        self.cam.distance = 8.0
        self.cam.elevation = -45.0

        mj.set_mjcb_control(self.controller)

    def controller(self, model, data):
        """
        This controller adds drag force to the ball
        The drag force has the form of
        F = (cv^Tv)v / ||v||
        """
        vx, vy, vz = self.data.qvel[0], self.data.qvel[1], self.data.qvel[2]
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        c = 1.0
        self.data.qfrc_applied[0] = -c * v * vx
        self.data.qfrc_applied[1] = -c * v * vy
        self.data.qfrc_applied[2] = -c * v * vz

    def simulate(self):
        with mj_viewer.launch_passive(self.model, self.data) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
            start = time.time()
            while viewer.is_running(): #and time.time() - start < 30:
                step_start = time.time()

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mj.mj_step(self.model, self.data)

                # Example modification of a viewer option: toggle contact points every two seconds.
                with viewer.lock():
                    viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.data.time % 2)
                
                # if i < end:
                #   pos = path[i]
                #   print(path[i]) 
                #   d.qpos[1] = pos
                # i = i + 1

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()


                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


def main():
    model_dir = Path("/home/anton/master/mujoco_menagerie/universal_robots_ur5e")
    model_xml = model_dir / "scene.xml"
    sim = Ur5e(str(model_xml))
    sim.reset()
    x = threading.Thread(target=sim.simulate)
    x.start()
    while True:
        print("Hej")
        time.sleep(1)


if __name__ == "__main__":
    main()
