import subprocess, math, time, sys, os, numpy as np
import matplotlib.pyplot as plt
import pybullet as bullet_simulation
import pybullet_data

# setup paths and load the core
abs_path = os.path.dirname(os.path.realpath(__file__))
root_path = abs_path + '/..'
core_path = root_path + '/core'
sys.path.append(core_path)
from Pybullet_Simulation import Simulation

# specific settings for this task

taskId = 3.2

try:
    if sys.argv[1] == 'nogui':
        gui = False
    else:
        gui = True
except:
    gui = True

pybulletConfigs = {
    "simulation": bullet_simulation,
    "pybullet_extra_data": pybullet_data,
    "gui": gui,
    "panels": False,
    "realTime": False,
    "controlFrequency": 1000,
    "updateFrequency": 250,
    "gravity": -9.81,
    "gravityCompensation": 1.,
    "floor": True,
    "cameraSettings": (1.2, 90, -22.8, (-0.12, -0.01, 0.99))
}
robotConfigs = {
    "robotPath": core_path + "/nextagea_description/urdf/NextageaOpen.urdf",
    "robotPIDConfigs": core_path + "/PD_gains.yaml",
    "robotStartPos": [0, 0, 0.85],
    "robotStartOrientation": [0, 0, 0, 1],
    "fixedBase": False,
    "colored": False
}

sim = Simulation(pybulletConfigs, robotConfigs)

##### Please leave this function unchanged, feel free to modify others #####
def getReadyForTask():
    global finalTargetPos
    global taleId, cubeId, targetId, obstacle
    finalTargetPos = np.array([0.35,0.38,1.0])
    # compile target urdf
    urdf_compiler_path = core_path + "/urdf_compiler.py"
    subprocess.call([urdf_compiler_path,
                     "-o", abs_path+"/lib/task_urdfs/task3_2_target_compiled.urdf",
                     abs_path+"/lib/task_urdfs/task3_2_target.urdf"])

    sim.p.resetJointState(bodyUniqueId=1, jointIndex=12, targetValue=-0.4)
    sim.p.resetJointState(bodyUniqueId=1, jointIndex=6, targetValue=-0.4)

    # load the table in front of the robot
    tableId = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/table/table_taller.urdf",
        basePosition          = [0.8, 0, 0],             
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,math.pi/2]),                                  
        useFixedBase          = True,             
        globalScaling         = 1.4
    )
    cubeId = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/cubes/task3_2_dumb_bell.urdf", 
        basePosition          = [0.5, 0, 1.1],            
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,0]),                                  
        useFixedBase          = False,             
        globalScaling         = 1.4
    )
    sim.p.resetVisualShapeData(cubeId, -1, rgbaColor=[1,1,0,1])
    
    targetId = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/task3_2_target_compiled.urdf",
        basePosition          = finalTargetPos,             
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,math.pi/4]), 
        useFixedBase          = True,             
        globalScaling         = 1
    )
    obstacle = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/cubes/task3_2_obstacle.urdf",
        basePosition          = [0.43,0.275,0.9],             
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,math.pi/4]), 
        useFixedBase          = True,             
        globalScaling         = 1
    )

    for _ in range(300):
        sim.tick()
        time.sleep(1./1000)

    return tableId, cubeId, targetId


def solution():
    print(sim.getJointPosition('LHAND'))
    print(sim.getJointPosition('RHAND'))

    sim.moveBothHands(np.array([0.42, 0.2, 1.0]),np.array([0, -1, 0]), np.array([0.42, 0.12, 1.0]),np.array([0, 1, 0]),speed=0.1)

    sim.moveBothHands(np.array([0.45, 0.2, 1.2]),np.array([0, -1, 0]), np.array([0.42, 0.1, 1.2]),np.array([0, 1, 0]),speed=0.1)
    
    sim.moveJointInConjuction('CHEST_JOINT0', np.deg2rad(50), angularSpeed=0.01)
    print(sim.getJointPosition('LHAND'))
    print(sim.getJointPosition('RHAND'))
    sim.moveBothHands(np.array([0.2, 0.55, 1.0]),np.array([0, 0, 0]), np.array([0.28, 0.45, 1.0]),np.array([0, 0, 0]),speed=0.1)
    sim.moveBothHands(np.array([0.2, 0.55, 1.0]),np.array([0, 0, 0]), np.array([0.35, 0.35, 1.0]),np.array([0, 0, 0]),speed=0.1)

    print(sim.getJointPosition('LHAND'))
    print(sim.getJointPosition('RHAND'))

tableId, cubeId, targetId = getReadyForTask()
solution()

print("Cube position:")
cubePosition=sim.p.getBasePositionAndOrientation(cubeId)

print(cubePosition)

print("Target position:")
targetPosition=sim.p.getBasePositionAndOrientation(targetId)
print(targetPosition)

print("Difference in position:")
diff=sim.getVectorLength(np.array(cubePosition[0])-np.array(targetPosition[0]))
print(diff)