from scipy.spatial.transform import Rotation as npRotation
from scipy.special import comb
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
import math
import re
import time
import yaml

from Pybullet_Simulation_base import Simulation_base

class Simulation(Simulation_base):
    """A Bullet simulation involving Nextage robot"""

    def __init__(self, pybulletConfigs, robotConfigs, refVect=None):
        """Constructor
        Creates a simulation instance with Nextage robot.
        For the keyword arguments, please see in the Pybullet_Simulation_base.py
        """
        super().__init__(pybulletConfigs, robotConfigs)
        if refVect:
            self.refVector = np.array(refVect)
        else:
            self.refVector = np.array([1,0,0])

    ########## Task 1: Kinematics ##########
    # Task 1.1 Forward Kinematics
    jointRotationAxis = {
        'base_to_dummy': np.zeros(3),  # Virtual joint
        'base_to_waist': np.zeros(3),  # Fixed joint
        # TODO: modify from here
        'CHEST_JOINT0': np.array([0, 0, 1]),
        'HEAD_JOINT0': np.array([0, 0, 1]),
        'HEAD_JOINT1': np.array([0, 1, 0]),
        'LARM_JOINT0': np.array([0, 0, 1]),
        'LARM_JOINT1': np.array([0, 1, 0]),
        'LARM_JOINT2': np.array([0, 1, 0]),
        'LARM_JOINT3': np.array([1, 0, 0]),
        'LARM_JOINT4': np.array([0, 1, 0]),
        'LARM_JOINT5': np.array([0, 0, 1]),
        'RARM_JOINT0': np.array([0, 0, 1]),
        'RARM_JOINT1': np.array([0, 1, 0]),
        'RARM_JOINT2': np.array([0, 1, 0]),
        'RARM_JOINT3': np.array([1, 0, 0]),
        'RARM_JOINT4': np.array([0, 1, 0]),
        'RARM_JOINT5': np.array([0, 0, 1]),
        'RHAND'      : np.array([0, 0, 0]),
        'LHAND'      : np.array([0, 0, 0])
    }

    frameTranslationFromParent = {
        'base_to_dummy': np.zeros(3),  # Virtual joint
        'base_to_waist': np.zeros(3),  # Fixed joint
        # TODO: modify from here
        'CHEST_JOINT0': np.array([0, 0, 0.267]),
        'HEAD_JOINT0': np.array([0, 0, 0.302]),
        'HEAD_JOINT1': np.array([0, 0, 0.066]),
        'LARM_JOINT0': np.array([0.04, 0.135, 0.1015]),
        'LARM_JOINT1': np.array([0, 0, 0.066]),
        'LARM_JOINT2': np.array([0, 0.095, -0.25]),
        'LARM_JOINT3': np.array([0.1805, 0, -0.03]),
        'LARM_JOINT4': np.array([0.1495, 0, 0]),
        'LARM_JOINT5': np.array([0, 0, -0.1335]),
        'RARM_JOINT0': np.array([0.04, -0.135, 0.1015]),
        'RARM_JOINT1': np.array([0, 0, 0.066]),
        'RARM_JOINT2': np.array([0, 0.095, -0.25]),
        'RARM_JOINT3': np.array([0.1805, 0, -0.03]),
        'RARM_JOINT4': np.array([0.1495, 0, 0]),
        'RARM_JOINT5': np.array([0, 0, -0.1335]),
        'RHAND'      : np.array([0, 0, 0]), # optional
        'LHAND'      : np.array([0, 0, 0]) # optional
    }

    jointParents = {
        'base_to_waist': 'base_to_dummy', 
        'CHEST_JOINT0': 'base_to_waist',
        'HEAD_JOINT0': 'CHEST_JOINT0',
        'HEAD_JOINT1': 'HEAD_JOINT0',
        'LARM_JOINT0': 'CHEST_JOINT0',
        'LARM_JOINT1': 'LARM_JOINT0',
        'LARM_JOINT2': 'LARM_JOINT1',
        'LARM_JOINT3': 'LARM_JOINT2',
        'LARM_JOINT4': 'LARM_JOINT3',
        'LARM_JOINT5': 'LARM_JOINT4',
        'RARM_JOINT0': 'CHEST_JOINT0',
        'RARM_JOINT1': 'RARM_JOINT0',
        'RARM_JOINT2': 'RARM_JOINT1',
        'RARM_JOINT3': 'RARM_JOINT2',
        'RARM_JOINT4': 'RARM_JOINT3',
        'RARM_JOINT5': 'RARM_JOINT4',
        'RHAND'      : 'RARM_JOINT5', 
        'LHAND'      : 'LARM_JOINT5'
    }

    def getJointRotationalMatrix(self, jointName=None, theta=None):
        """
            Returns the 3x3 rotation matrix for a joint from the axis-angle representation,
            where the axis is given by the revolution axis of the joint and the angle is theta.
            NOTE theta currently taken in rad
        """
        if jointName == None:
            raise Exception("[getJointRotationalMatrix] \
                Must provide a joint in order to compute the rotational matrix!")
        # TODO modify from here
        # Hint: the output should be a 3x3 rotational matrix as a numpy array
        #return np.matrix()
        if theta == None:
            # NOTE if this breaks change it to the commented version
            raise Exception("[getJointRotationalMatrix] \
                Must provide a rotation angle in order to compute the rotational matrix!")
            # return np.matrix([
            #                 [1,0,0],
            #                 [0,1,0],
            #                 [0,0,1]
            #               ])
        
        else:
            rotation_axis = self.jointRotationAxis[jointName]

            if rotation_axis == np.array([1, 0, 0]):
                return np.matrix([
                            [1,0,0],
                            [0,np.cos(np.deg2rad(theta)),-np.sin(np.deg2rad(theta))],
                            [0,np.sin(np.deg2rad(theta)),np.cos(np.deg2rad(theta))]
                        ])
            elif rotation_axis == np.array([0, 1, 0]):
                ##  TODO investigate sign reversal thing Ana mentioned
                return np.matrix([
                            [np.cos(np.deg2rad(theta)),0,np.sin(np.deg2rad(theta))],
                            [0,1,0],
                            [-np.sin(np.deg2rad(theta)),0,np.cos(np.deg2rad(theta))]
                        ])
            elif rotation_axis == np.array([0, 0, 1]):
                return np.matrix([
                            [np.cos(np.deg2rad(theta)),-np.sin(np.deg2rad(theta)),0],
                            [np.sin(np.deg2rad(theta)),np.cos(np.deg2rad(theta)),0],
                            [0,0,1]
                        ])
            else:
                return np.matrix(np.identity(3))
            
    def getTransformationMatrices(self, thetasDict=None):
        """
            Returns the homogeneous transformation matrices for each joint as a dictionary of matrices.
            Currently returns them relative to their parents.
        """
        transformationMatrices = {}
        # TODO modify from here
        # Hint: the output should be a dictionary with joint names as keys and
        # their corresponding homogeneous transformation matrices as values.
        
        

        for jointName in self.frameTranslationFromParent:
            if thetasDict and jointName in thetasDict:
                rotMatrix = self.getJointRotationalMatrix(jointName, theta=thetasDict[jointName])
            else:    
                rotMatrix = self.getJointRotationalMatrix(jointName, self.getJointPos(jointName))

            # Stacking the rotation matrix and translation vector
            nonAugmentedTransformationMatrix = np.hstack((rotMatrix, np.reshape(self.frameTranslationFromParent[jointName], (3,1))))

            # adding the augmentation row and setting the matrix
            transformationMatrices[jointName] = np.vstack((nonAugmentedTransformationMatrix,[0,0,0,1]))
            
        return transformationMatrices

    def getJointLocationAndOrientation(self, jointName, thetasDict=None):
        """
            Returns the position and rotation matrix of a given joint using Forward Kinematics
            according to the topology of the Nextage robot.
        """
        # Remember to multiply the transformation matrices following the kinematic chain for each arm.
        #TODO modify from here
        # Hint: return two numpy arrays, a 3x1 array for the position vector,
        # and a 3x3 array for the rotation matrix
        #return pos, rotmat
        transformationMatrices = self.getTransformationMatrices(thetasDict)
    
        if jointName == 'base_to_dummy' or 'base_to_waist':
            return transformationMatrices[jointName]
        
        fkMatrix = transformationMatrices[jointName]
        currJoint = self.jointParents[jointName]

        while currJoint != 'base_to_waist':
            fkMatrix = transformationMatrices[currJoint] @ fkMatrix
            currJoint = self.jointParents[currJoint]

        return np.array(fkMatrix[:3, 3].reshape((1,3))), np.array(fkMatrix[:3,:3])

    def getJointPosition(self, jointName):
        """Get the position of a joint in the world frame, leave this unchanged please."""
        return self.getJointLocationAndOrientation(jointName)[0]

    def getJointOrientation(self, jointName, ref=None):
        """Get the orientation of a joint in the world frame, leave this unchanged please."""
        if ref is None:
            return np.array(self.getJointLocationAndOrientation(jointName)[1] @ self.refVector).squeeze()
        else:
            return np.array(self.getJointLocationAndOrientation(jointName)[1] @ ref).squeeze()

    def getJointAxis(self, jointName):
        """Get the orientation of a joint in the world frame, leave this unchanged please."""
        return np.array(self.getJointLocationAndOrientation(jointName)[1] @ self.jointRotationAxis[jointName]).squeeze()

# def geomJacobian(jnt2pos, jnt3pos, endEffPos):
    
#     ai = np.array([0,0,1])
#     col0 = np.array(endEffPos + [0])
#     col1 = np.array(endEffPos + [0]) - np.array(jnt2pos + [0])
#     col2 = np.array(endEffPos + [0]) - np.array(jnt3pos + [0])
#     J = np.array([np.cross(ai,col0), np.cross(ai,col1), np.cross(ai,col2)]).T 
#     return J

    def jacobianMatrix(self, endEffector, relevantJoints=None):
        """Calculate the Jacobian Matrix for the Nextage Robot."""
        # TODO modify from here
        # You can implement the cross product yourself or use calculateJacobian().
        # Hint: you should return a numpy array for your Jacobian matrix. The
        # size of the matrix will depend on your chosen convention. You can have
        # a 3xn or a 6xn Jacobian matrix, where 'n' is the number of joints in
        # your kinematic chain.
        #return np.array()

        if not relevantJoints:
            relevantJoints = self.getRelevantJoints(endEffector=endEffector)

        Jpos = np.zeros((3,len(relevantJoints)))
        Jvec = np.zeros((3,len(relevantJoints)))

        endEffPos = self.getJointPosition(endEffector)
        endEffAxis = self.getJointOrientation(endEffector)

        for i in range(len(relevantJoints)):
            currJoint = relevantJoints[i]
    
            jointPos = self.getJointPosition(currJoint)
            jointAxis = self.getJointAxis(currJoint)

            Jpos[:,i] = np.cross(jointAxis, (endEffPos - jointPos))
            Jvec[:, i] = np.cross(jointAxis, endEffAxis)

        return np.vstack((Jpos, Jvec))

    def calcJacobianCustomAngles(self, endEffPos, endEffOrientation, jointAngles):
        
        Jpos = np.zeros((3,len(jointAngles)))
        Jvec = np.zeros((3,len(jointAngles)))

        i = 0
        for currJoint in jointAngles:

            jointPos, jointRotationMatrix = self.getJointLocationAndOrientation(currJoint, jointAngles)
            jointAxis = np.squeeze(jointRotationMatrix @ self.jointRotationAxis[currJoint])

            Jpos[:,i] = np.cross(jointAxis, (endEffPos - jointPos))
            Jvec[:, i] = np.cross(jointAxis, endEffOrientation)

            i += 1

        return np.vstack((Jpos, Jvec))


    def getRelevantJoints(self, endEffector):
        reversedChain = []
        
        currJoint = self.jointParents[endEffector]
        
        while currJoint != 'base_to_waist':
            reversedChain.append(currJoint)
            currJoint = self.jointParents[currJoint]
        
        reversedChain.reverse()

        return reversedChain

    def getJointAngles(self, jointNames):
        return list(map(self.getJointPos,jointNames))

    # Task 1.2 Inverse Kinematics

    # NOTE documented "Keywork" arguments will just be kept in frame and ignored for now 
    def inverseKinematics(self, endEffector, targetPosition, orientation, frame=None):
        """Your IK solver \\
        Arguments: \\
            endEffector: the jointName the end-effector \\
            targetPosition: final destination the the end-effector \\
            orientation: the desired orientation of the end-effector
                        together with its parent link \\
        Keywork Arguments: \\
            speed: how fast the end-effector should move (m/s) \\
            orientation: the desired orientation \\
            compensationRatio: naive gravity compensation ratio \\
            debugLine: optional \\
            verbose: optional \\
        Return: \\
            Vector of x_refs
        """

        relevantJoints = self.getRelevantJoints(endEffector)

        numSteps = 0
        if "steps" in frame:
            numSteps = frame["steps"]
        else:
            numSteps = 100

        start_pos, start_orientation = self.getJointLocationAndOrientation(endEffector)

        xRefs = np.zeros((len(relevantJoints),numSteps))

        xRefs[:, 0] = self.getJointAngles(relevantJoints)

        for i in range(1,numSteps):
            thetasDict = { jointName:jointAngle for (jointName, jointAngle) in zip(relevantJoints, xRefs[:, i-1])}
            currEndEffPos, currEndEffectorRotationMatrix = self.getJointLocationAndOrientation(endEffector, thetasDict)

            currJacobian = self.calcJacobianCustomAngles(currEndEffPos, currEndEffectorRotationMatrix @ self.refVector, thetasDict)
            nextPosition, nextOrientation = start_pos + (i/numSteps)*(targetPosition - start_pos), start_orientation + (i/numSteps)*(orientation - start_orientation)
            xRefs[:, i] = xRefs[:, i-1] + np.linalg.pinv(currJacobian) @ ((np.hstack(nextPosition, nextOrientation)) - np.hstack((currEndEffPos,currEndEffectorRotationMatrix @ self.refVector)))
        # TODO add your code here
        # Hint: return a numpy array which includes the reference angular
        # positions for all joints after performing inverse kinematics.
        return xRefs

    def move_without_PD(self, endEffector, targetPosition, speed=0.01, orientation=None,
        threshold=1e-3, maxIter=3000, debug=False, verbose=False):
        """
        Move joints using Inverse Kinematics solver (without using PD control).
        This method should update joint states directly.
        Return:
            pltTime, pltDistance arrays used for plotting
        """
        #TODO add your code here
        # iterate through joints and update joint states based on IK solver
        relevantJoints = self.getRelevantJoints(endEffector)

        start_pos = self.getJointPosition(endEffector)

        numSteps = (targetPosition-start_pos)/speed
        numSteps = min(numSteps, maxIter)

        jointSteps = self.inverseKinematics(endEffector, targetPosition, orientation, {"steps": numSteps})

        pltTime = []
        pltDistance = []
        for i in range(jointSteps.shape[1]):
            jointConfigs = jointSteps[:,i]

            for jointName,angle in zip(relevantJoints,jointConfigs):
                self.p.resetJointState(
                    self.robot, self.jointIds[jointName], angle)

            currEndEffPos = self.getJointPosition(endEffector)
            pltTime.append(i)
            pltDistance.append(currEndEffPos - start_pos)

        return pltTime, pltDistance

    def tick_without_PD(self):
        """Ticks one step of simulation without PD control. """
        # TODO modify from here
        # Iterate through all joints and update joint states.
            # For each joint, you can use the shared variable self.jointTargetPos.

        self.p.stepSimulation()
        self.drawDebugLines()
        time.sleep(self.dt)


    ########## Task 2: Dynamics ##########
    # Task 2.1 PD Controller
    def calculateTorque(self, x_ref, x_real, dx_ref, dx_real, integral, kp, ki, kd):
        """ This method implements the closed-loop control \\
        Arguments: \\
            x_ref - the target position \\
            x_real - current position \\
            dx_ref - target velocity \\
            dx_real - current velocity \\
            integral - integral term (set to 0 for PD control) \\
            kp - proportional gain \\
            kd - derivetive gain \\
            ki - integral gain \\
        Returns: \\
            u(t) - the manipulation signal
        """
        # TODO: Add your code here
        pass

    # Task 2.2 Joint Manipulation
    def moveJoint(self, joint, targetPosition, targetVelocity, verbose=False):
        """ This method moves a joint with your PD controller. \\
        Arguments: \\
            joint - the name of the joint \\
            targetPos - target joint position \\
            targetVel - target joint velocity
        """
        def toy_tick(x_ref, x_real, dx_ref, dx_real, integral):
            # loads your PID gains
            jointController = self.jointControllers[joint]
            kp = self.ctrlConfig[jointController]['pid']['p']
            ki = self.ctrlConfig[jointController]['pid']['i']
            kd = self.ctrlConfig[jointController]['pid']['d']

            ### Start your code here: ###
            # Calculate the torque with the above method you've made
            torque = 0.0
            ### To here ###

            pltTorque.append(torque)

            # send the manipulation signal to the joint
            self.p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=self.jointIds[joint],
                controlMode=self.p.TORQUE_CONTROL,
                force=torque
            )
            # calculate the physics and update the world
            self.p.stepSimulation()
            time.sleep(self.dt)

        targetPosition, targetVelocity = float(targetPosition), float(targetVelocity)

        # disable joint velocity controller before apply a torque
        self.disableVelocityController(joint)
        # logging for the graph
        pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity = [], [], [], [], [], []

        return pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity

    def move_with_PD(self, endEffector, targetPosition, speed=0.01, orientation=None,
        threshold=1e-3, maxIter=3000, debug=False, verbose=False):
        """
        Move joints using inverse kinematics solver and using PD control.
        This method should update joint states using the torque output from the PD controller.
        Return:
            pltTime, pltDistance arrays used for plotting
        """
        #TODO add your code here
        # Iterate through joints and use states from IK solver as reference states in PD controller.
        # Perform iterations to track reference states using PD controller until reaching
        # max iterations or position threshold.

        # Hint: here you can add extra steps if you want to allow your PD
        # controller to converge to the final target position after performing
        # all IK iterations (optional).

        #return pltTime, pltDistance
        pass

    def tick(self):
        """Ticks one step of simulation using PD control."""
        # Iterate through all joints and update joint states using PD control.
        for joint in self.joints:
            # skip dummy joints (world to base joint)
            jointController = self.jointControllers[joint]
            if jointController == 'SKIP_THIS_JOINT':
                continue

            # disable joint velocity controller before apply a torque
            self.disableVelocityController(joint)

            # loads your PID gains
            kp = self.ctrlConfig[jointController]['pid']['p']
            ki = self.ctrlConfig[jointController]['pid']['i']
            kd = self.ctrlConfig[jointController]['pid']['d']

            ### Implement your code from here ... ###
            # TODO: obtain torque from PD controller
            torque = 0.0  # TODO: fix me
            ### ... to here ###

            self.p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=self.jointIds[joint],
                controlMode=self.p.TORQUE_CONTROL,
                force=torque
            )

            # Gravity compensation
            # A naive gravitiy compensation is provided for you
            # If you have embeded a better compensation, feel free to modify
            compensation = self.jointGravCompensation[joint]
            self.p.applyExternalForce(
                objectUniqueId=self.robot,
                linkIndex=self.jointIds[joint],
                forceObj=[0, 0, -compensation],
                posObj=self.getLinkCoM(joint),
                flags=self.p.WORLD_FRAME
            )
            # Gravity compensation ends here

        self.p.stepSimulation()
        self.drawDebugLines()
        time.sleep(self.dt)

    ########## Task 3: Robot Manipulation ##########
    def cubic_interpolation(self, points, nTimes=100):
        """
        Given a set of control points, return the
        cubic spline defined by the control points,
        sampled nTimes along the curve.
        """
        #TODO add your code here
        # Return 'nTimes' points per dimension in 'points' (typically a 2xN array),
        # sampled from a cubic spline defined by 'points' and a boundary condition.
        # You may use methods found in scipy.interpolate

        #return xpoints, ypoints
        pass

    # Task 3.1 Pushing
    def dockingToPosition(self, leftTargetAngle, rightTargetAngle, angularSpeed=0.005,
            threshold=1e-1, maxIter=300, verbose=False):
        """A template function for you, you are free to use anything else"""
        # TODO: Append your code here
        pass

    # Task 3.2 Grasping & Docking
    def clamp(self, leftTargetAngle, rightTargetAngle, angularSpeed=0.005, threshold=1e-1, maxIter=300, verbose=False):
        """A template function for you, you are free to use anything else"""
        # TODO: Append your code here
        pass

 ### END
