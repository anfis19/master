from spatialmath import SE3
import roboticstoolbox as rtb
robot = rtb.models.UR5()



Tep = SE3.Trans(0.6, -0.3, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
sol = robot.ik_LM(Tep)         # solve IK
print(Tep)

q_pickup = sol[0]
print(robot.fkine(q_pickup)) 