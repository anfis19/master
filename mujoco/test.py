import mujoco as mj
from pathlib import Path
import numpy as np
import glfw as g

class MujocoSimulation:
    def __init__(self):
        # Load model, initialize variables
        self.dragging = False
        self.mouse_x, self.mouse_y = None, None
        self.attachment_point = None
        self.force = np.zeros(3)
        # ...
# GLFW callback functions
    def on_mouse_press(self, window, button, action, mods):
        if button == mj.glfw.glfw.MOUSE_BUTTON_LEFT and action == mj.glfw.glfw.PRESS:
            # Get mouse click position and identify attachment point on robot
            self.mouse_x, self.mouse_y = mj.glfw.glfw.get_cursor_pos(window)
            # Replace this with your logic to identify the clicked point on the robot's body (e.g., using raycasting)
            self.attachment_point = None  # Placeholder, replace with actual attachment point
            self.dragging = True
        print('WFT')


    def on_mouse_release(self, window, button, action, mods):
        if button == mj.glfw.glfw.MOUSE_BUTTON_LEFT and action == mj.glfw.glfw.RELEASE:
            self.dragging = False
        if button == mj.glfw.glfw.MOUSE_BUTTON_LEFT and action == mj.glfw.glfw.PRESS:
            # Get mouse click position and identify attachment point on robot
            self.mouse_x, self.mouse_y = mj.glfw.glfw.get_cursor_pos(window)
            # Replace this with your logic to identify the clicked point on the robot's body (e.g., using raycasting)
            self.attachment_point = None  # Placeholder, replace with actual attachment point
            self.dragging = True

    def on_mouse_motion(self, window, xpos, ypos):
        # print('...')
        if self.dragging:
            # Calculate mouse movement relative to click position
            dx = xpos - self.mouse_x
            dy = ypos - self.mouse_y
            # Convert movement to force (replace with your desired force calculation)
            self.force = np.array([dx, dy, 0])  # Adjust the force based on your needs
            # Update mouse position for next movement calculation
            self.mouse_x, self.mouse_y = xpos, ypos

def main():
    max_width = 100
    max_height = 100
    #ctx = mj.GLContext(max_width, max_height)
    #ctx.make_current()
    
    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    sim = MujocoSimulation()
    mj.glfw.glfw.init()
    window = mj.glfw.glfw.create_window(1200, 900, "Demo", None, None)
    mj.glfw.glfw.set_cursor_pos_callback(window, sim.on_mouse_motion)
    mj.glfw.glfw.set_mouse_button_callback(window, sim.on_mouse_release)

    mj.glfw.glfw.make_context_current(window)
    mj.glfw.glfw.swap_interval(1)

    mj.mjv_defaultCamera(cam)
    mj.mjv_defaultOption(opt)

    model_dir = Path("/home/anton/master/mujoco_menagerie/universal_robots_ur5e")
    model_xml = model_dir / "scene.xml"


    
    model = mj.MjModel.from_xml_path(str(model_xml))
    data = mj.MjData(model)

    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    count = 0


    while not mj.glfw.glfw.window_should_close(window):
        simstart = data.time

        while (data.time - simstart < 1.0/60.0):
            mj.mj_step(model, data)

        #viewport = mj.MjrRect(0, 0, 0, 0)
        #mj.glfw.glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, 1200, 900)

        if sim.dragging:
            print('Yay' + str(count))
            count = count + 1

        #mj.mjv_updateScene(model, data, opt, None, cam, 0, scene)
        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)

        mj.glfw.glfw.swap_buffers(window)
        mj.glfw.glfw.poll_events()

    mj.glfw.glfw.terminate()


if __name__ == "__main__":
    main()