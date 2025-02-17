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
from time import sleep
from scipy.interpolate import CubicSpline

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
        'CHEST_JOINT0': np.array([0, 0, 1.117]),
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

    # defines the parent of each joint in the robot, useful for computing the kinematic chain
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
     
        if theta == None:
            raise Exception("[getJointRotationalMatrix] \
                Must provide a rotation angle in order to compute the rotational matrix!")
        
        else:
            rotation_axis = self.jointRotationAxis[jointName]

            if np.array_equal(rotation_axis, np.array([1, 0, 0])):
                return np.matrix([
                            [1,0,0],
                            [0,np.cos(theta),-np.sin(theta)],
                            [0,np.sin(theta),np.cos(theta)]
                        ])
            elif np.array_equal(rotation_axis, np.array([0, 1, 0])):
                return np.matrix([
                            [np.cos(theta),0,np.sin(theta)],
                            [0,1,0],
                            [-np.sin(theta),0,np.cos(theta)]
                        ])
            elif np.array_equal(rotation_axis, np.array([0, 0, 1])):
                return np.matrix([
                            [np.cos(theta),-np.sin(theta),0],
                            [np.sin(theta),np.cos(theta),0],
                            [0,0,1]
                        ])
            else:
                return np.matrix(np.identity(3))
            
    # accepts an optional "thetasDict" parameter for custom joint configurations (such as for a trajectory being generated)
    def getTransformationMatrices(self, thetasDict=None):
        """
            Returns the homogeneous transformation matrices for each joint as a dictionary of matrices.
            Currently returns them relative to their parents.
        """
        transformationMatrices = {}  

        for jointName in self.frameTranslationFromParent:
            if jointName not in ['RHAND', 'LHAND']:
                if thetasDict and jointName in thetasDict:
                    rotMatrix = self.getJointRotationalMatrix(jointName, theta=thetasDict[jointName])
                else:    
                    rotMatrix = self.getJointRotationalMatrix(jointName, self.getJointPos(jointName))
            else:
                rotMatrix = np.identity(3)

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

        if jointName == 'RHAND' or jointName == 'LHAND':
            jointName = self.jointParents[jointName]

        transformationMatrices = self.getTransformationMatrices(thetasDict)

        if jointName in ['base_to_dummy','base_to_waist']:
            raise Exception("Shouldn't be calling this for dummy joints!")

        fkMatrix = transformationMatrices[jointName]
        currJoint = self.jointParents[jointName]

        # iterating back through the kinematic chain and multiplying the HTMs together
        numMatricesIncluded = 1
        while currJoint != 'base_to_waist':
            fkMatrix = transformationMatrices[currJoint] @ fkMatrix
            numMatricesIncluded += 1
            currJoint = self.jointParents[currJoint]

        # sanity check
        assert numMatricesIncluded == len(self.getRelevantJoints(jointName))
        
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

    def jacobianMatrix(self, endEffector, relevantJoints=None, endEffectorPosition=None, endEffectorOrientation=None):
        """Calculate the Jacobian Matrix for the Nextage Robot."""

        if not relevantJoints:
            relevantJoints = self.getRelevantJoints(endEffector=endEffector)

        Jpos = np.zeros((3,len(relevantJoints)))
        Jvec = np.zeros((3,len(relevantJoints)))

        if not np.any(endEffectorPosition):
            endEffPos = self.getJointPosition(endEffector)
        else:
            endEffPos = endEffectorPosition

        if not np.any(endEffectorOrientation):
            endEffAxis = self.getJointOrientation(endEffector)
        else:
            endEffAxis = endEffectorOrientation

        for i in range(len(relevantJoints)):
            currJoint = relevantJoints[i]
    
            jointPos = self.getJointPosition(currJoint)
            jointAxis = self.getJointAxis(currJoint)

            Jpos[:,i] = np.cross(jointAxis, (endEffPos - jointPos))
            Jvec[:, i] = np.cross(jointAxis, endEffAxis)

        return np.vstack((Jpos, Jvec))

    # jacobian implementation which accepts custom joint configurations
    def calcJacobianCustomAngles(self, endEffPos, endEffOrientation=None, jointAngles={}):
        
        Jpos = np.zeros((3,len(jointAngles)))
        Jvec = np.zeros((3,len(jointAngles)))

        i = 0
        for currJoint in jointAngles:

            jointPos, jointRotationMatrix = self.getJointLocationAndOrientation(currJoint, jointAngles)
            jointAxis = np.squeeze(jointRotationMatrix @ self.jointRotationAxis[currJoint])

            Jpos[:,i] = np.cross(jointAxis, (endEffPos - jointPos))
            if np.any(endEffOrientation):
                Jvec[:, i] = np.cross(jointAxis, endEffOrientation)

            i += 1

        if np.any(endEffOrientation):
            return np.vstack((Jpos, Jvec))
        else:
            return Jpos

    # gets the joints in the kinematic chain of the passed in end effector (or joint)
    def getRelevantJoints(self, endEffector):
        reversedChain = []
        
        if endEffector in ['LHAND', 'RHAND']:
            currJoint = self.jointParents[endEffector]
        else:
            currJoint = endEffector

        while currJoint != 'base_to_waist':
            reversedChain.append(currJoint)
            currJoint = self.jointParents[currJoint]
        
        reversedChain.reverse()
        return reversedChain

    # returns a list of the joints' current configurations
    def getJointAngles(self, jointNames):
        return list(map(self.getJointPos,jointNames))

    # Task 1.2 Inverse Kinematics

    # implementation of inverseKinematics which computes the target joint configurations in a single step
    def singleStepInverseKinematics(self, endEffector, targetPosition, orientation, frame=None):
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
        assert "relevantJoints" in frame

        relevantJoints = frame.get('relevantJoints')
        start_pos, start_orientation = self.getJointLocationAndOrientation(endEffector)

        start_orientation = start_orientation @ self.refVector
        xRefDeltas = np.zeros(len(relevantJoints))
        # print(currEndEffPos)
        currJacobian = self.jacobianMatrix(endEffector, relevantJoints, endEffectorPosition=start_pos, endEffectorOrientation=start_orientation)

        if np.any(orientation):
            posOrientationDelta = ((np.hstack((np.squeeze(targetPosition), orientation))) - np.hstack((np.squeeze(start_pos),start_orientation)))
            xRefDeltas = np.linalg.pinv(currJacobian) @ posOrientationDelta
            # xRefDeltas = np.linalg.pinv(currJacobian) @ ((np.hstack((targetPosition, orientation))) - np.hstack((start_pos,start_orientation @ self.refVector)))
        else:
            xRefDeltas = np.linalg.pinv(currJacobian[:3]) @ np.squeeze(targetPosition - start_pos)
            
        xRefs = np.array(self.getJointAngles(relevantJoints)) + xRefDeltas

        return xRefs

    # IK implementation which divides the trajectory into a number of steps determined by the passed in speed (in the frame)
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
        assert "relevantJoints" in frame, "relevantJoints missing from frame in IK function call"
        assert "steps" in frame, "steps missing from frame in IK function call"
        
        relevantJoints = frame['relevantJoints']
        numSteps = frame['steps']


        start_pos = self.getJointPosition(endEffector)
        start_orientation = self.getJointOrientation(endEffector)
        
        xRefs = np.zeros((len(relevantJoints),numSteps+1))

        xRefs[:, 0] = self.getJointAngles(relevantJoints)

        for i in range(1,numSteps+1):
            thetasDict = { jointName:jointAngle for (jointName, jointAngle) in zip(relevantJoints, xRefs[:, i-1])}
            currEndEffPos, currEndEffectorRotationMatrix = self.getJointLocationAndOrientation(endEffector, thetasDict)

            currOrientation = np.squeeze(currEndEffectorRotationMatrix @ self.refVector)

            currJacobian = self.calcJacobianCustomAngles(currEndEffPos, endEffOrientation=currOrientation, jointAngles=thetasDict)
            nextPosition = start_pos + (i/numSteps)*(targetPosition - start_pos)

            if np.any(orientation):
                nextOrientation = start_orientation + (i/numSteps)*(orientation - start_orientation)

                posOrientationDelta = ((np.hstack((np.squeeze(nextPosition), nextOrientation))) - np.hstack((np.squeeze(currEndEffPos),currOrientation)))
                xRefDeltas = np.linalg.pinv(currJacobian) @ posOrientationDelta
                xRefs[:, i] = xRefs[:, i-1] + xRefDeltas
            else:
                xRefs[:, i] = xRefs[:, i-1] + np.linalg.pinv(currJacobian[:3]) @ np.squeeze(nextPosition - currEndEffPos)
        
        return xRefs

    def move_without_PD(self, endEffector, targetPosition, speed=0.01, orientation=None,
        threshold=1e-3, maxIter=3000, debug=False, verbose=False):
        """
        Move joints using Inverse Kinematics solver (without using PD control).
        This method should update joint states directly.
        Return:
            pltTime, pltDistance arrays used for plotting
        """
        relevantJoints = self.getRelevantJoints(endEffector)

        start_pos = self.getJointPosition(endEffector).squeeze()
        
        # calculating the number of steps for the IK configurations
        numSteps = self.calIterToTarget(start_pos,targetPosition, speed)
        numSteps = min(numSteps, maxIter)

        jointSteps = self.inverseKinematics(endEffector, targetPosition, orientation, {"steps": numSteps, "relevantJoints": relevantJoints})

        pltTime = []
        pltDistance = []
        for i in range(jointSteps.shape[1]):
            jointConfigs = jointSteps[:,i]

            for jointName,angle in zip(relevantJoints,jointConfigs):
                self.p.resetJointState(
                    self.robot, self.jointIds[jointName], angle)

            # logging for the graph
            currEndEffPos = self.getJointPosition(endEffector)
            pltTime.append(i*self.dt)
            pltDistance.append(self.getVectorLength(targetPosition - currEndEffPos.squeeze()))
            self.tick_without_PD(relevantJoints, jointConfigs)

        return pltTime, pltDistance

    def tick_without_PD(self, relevantJoints=None, jointPositions=None):
        """Ticks one step of simulation without PD control. """

        assert relevantJoints is not None
        assert jointPositions is not None
        
        for jointName,angle in zip(relevantJoints,jointPositions):
                self.jointTargetPos[jointName] = angle

        self.p.stepSimulation()
        self.drawDebugLines()
        time.sleep(self.dt)

    # "On the fly" implementation of move_without_PD() which calls single step inverse kinematics at each iteration rather than calculating in advance
    def move_without_PD_on_the_fly(self, endEffector, targetPosition, speed=0.01, orientation=None,
        threshold=1e-3, maxIter=3000, debug=False, verbose=False):
        """
        Move joints using Inverse Kinematics solver (without using PD control).
        This method should update joint states directly.
        Return:
            pltTime, pltDistance arrays used for plotting
        """
        relevantJoints = self.getRelevantJoints(endEffector)

        start_pos = self.getJointPosition(endEffector)

        # print(targetPosition)
        # print(start_pos)
        numSteps = self.calIterToTarget(start_pos.squeeze(), targetPosition, speed)
        numSteps = min(numSteps, maxIter)
        print(numSteps)

        endEffPositionTrajectory = np.linspace(start_pos, targetPosition, numSteps)

        if np.any(orientation):
            startOrientation = self.getJointOrientation(endEffector)
            endEffOrientationTrajectory = np.linspace(startOrientation, orientation, numSteps)

        # jointSteps = self.inverseKinematics(endEffector, targetPosition, orientation, {"relevantJoints": relevantJoints})

        pltTime = []
        pltDistance = []
        for i in range(1,numSteps):
            
            nextPosition = endEffPositionTrajectory[i]

            if np.any(orientation):
                nextOrientation = endEffOrientationTrajectory[i]
            else:
                nextOrientation = None
            
            newXRefs = self.singleStepInverseKinematics(endEffector, nextPosition, nextOrientation, {"relevantJoints": relevantJoints})

            for jointName,angle in zip(relevantJoints,newXRefs):
                self.p.resetJointState(
                    self.robot, self.jointIds[jointName], angle)

            pltTime.append(i)
            pltDistance.append(np.linalg.norm(nextPosition - start_pos))
            self.tick_without_PD(relevantJoints, newXRefs)

        return pltTime, pltDistance

    ########## Task 2: Dynamics ##########
    # Task 2.1 PD Controller

    # implements a PID function, to use it as a PD controller with velocity "drag" instead of damping:
    #   1. set integral to 0
    #   2. set dx_ref to 0
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
        return kp*(x_ref - x_real) + ki*integral + kd*(dx_ref - dx_real)

    # Task 2.2 Joint Manipulation
    def moveJoint(self, joint, targetPosition, targetVelocity, verbose=False, numSeconds=2):
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
            torque = self.calculateTorque(x_ref,x_real, dx_ref, dx_real, integral, kp, ki, kd)
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

        startPosition = self.getJointPos(joint)

        # disable joint velocity controller before apply a torque
        self.disableVelocityController(joint)
        # logging for the graph
        pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity = [], [], [], [], [], []
        
        disableIntegralValue = 0

        for i in range(int(numSeconds/self.dt)):
            currentPosition = self.getJointPos(joint)

            pltPosition.append(currentPosition)
            pltTime.append(i*self.dt)
            pltTarget.append(targetPosition)

            if i == 0:
                currentVelocity = 0
            else:
                currentVelocity = (currentPosition - pltPosition[i-1])/self.dt

            pltVelocity.append(currentVelocity)
            pltTorqueTime.append(i*self.dt)

            toy_tick(targetPosition, currentPosition, targetVelocity, currentVelocity, disableIntegralValue)


        return pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity

    def move_with_PD(self, endEffector, targetPosition, speed=0.01, orientation=None,
        threshold=1e-3, maxIter=3000, debug=False, verbose=False):
        """
        Move joints using inverse kinematics solver and using PD control.
        This method should update joint states using the torque output from the PD controller.
        Return:
            pltTime, pltDistance arrays used for plotting
        """

        relevantJoints = self.getRelevantJoints(endEffector)

        start_pos = self.getJointPosition(endEffector).squeeze()
        
        numSteps = self.calIterToTarget(start_pos,targetPosition, speed)
        numSteps = min(numSteps, maxIter)

        # calculating joint configurations 
        jointRefs = self.inverseKinematics(endEffector, targetPosition, orientation, {"steps": numSteps, "relevantJoints": relevantJoints})
        
        # storing for velocity calculations in tick()
        jointPrevPositions = {jointName:jointPos for jointName,jointPos in zip(relevantJoints, self.getJointAngles(relevantJoints)) }

        pltTime = []
        pltDistance = []

        for i in range(jointRefs.shape[1]):
            jointConfigs = jointRefs[:,i]

            for jointName,angle in zip(relevantJoints,jointConfigs):
                self.jointTargetPos[jointName] = angle
            
            # currEndEffPos = self.getJointPosition(endEffector)
            # print(currEndEffPos)

            positionsBeforeTick = {jointName:jointPos for jointName,jointPos in zip(relevantJoints, self.getJointAngles(relevantJoints)) }
            
            self.tick(jointPrevPositions)

            # updating previous position
            jointPrevPositions = positionsBeforeTick
            
            # pltTime.append(i*self.dt)
            # pltDistance.append(np.linalg.norm(currEndEffPos - targetPosition))
        
        # currentPosition
        # # while (selftargetPosition - currentPosition
        # currEndEffPos = self.getJointPosition(endEffector)
        # i = jointRefs.shape[1]

        # while self.getVectorLength(targetPosition - currEndEffPos.squeeze()) > threshold:
        #     positionsBeforeTick = {jointName:jointPos for jointName,jointPos in zip(relevantJoints, self.getJointAngles(relevantJoints)) }

        #     self.tick(jointPrevPositions)
        #     jointPrevPositions = positionsBeforeTick
            
        #     pltTime.append(i*self.dt)
        #     pltDistance.append(np.linalg.norm(currEndEffPos - targetPosition))
        #     i += 1

        return pltTime, pltDistance
        
    # "On the fly" implementation of move_with_PD() which calls single step inverse kinematics at each iteration rather than calculating in advance
    def move_with_PD_on_the_fly(self, endEffector, targetPosition, speed=0.01, orientation=None,
        threshold=1e-3, maxIter=3000, debug=False, verbose=False):
        """
        Move joints using inverse kinematics solver and using PD control.
        This method should update joint states using the torque output from the PD controller.
        Return:
            pltTime, pltDistance arrays used for plotting
        """

        relevantJoints = self.getRelevantJoints(endEffector)

        start_pos = self.getJointPosition(endEffector).squeeze()
        
        numSteps = self.calIterToTarget(start_pos,targetPosition, speed)
        numSteps = min(numSteps, maxIter)

        endEffPositionTrajectory = np.linspace(start_pos, targetPosition, numSteps)


        if np.any(orientation):
            startOrientation = self.getJointOrientation(endEffector)
            endEffOrientationTrajectory = np.linspace(startOrientation, orientation, numSteps)


        
        jointPrevPositions = {jointName:jointPos for jointName,jointPos in zip(relevantJoints, self.getJointAngles(relevantJoints)) }

        for i in range(1,numSteps):
            
            nextPosition = endEffPositionTrajectory[i]

            if np.any(orientation):
                nextOrientation = endEffOrientationTrajectory[i]
            else:
                nextOrientation = None

            newXRefs = self.singleStepInverseKinematics(endEffector, nextPosition, nextOrientation, {"relevantJoints": relevantJoints})

            for jointName,angle in zip(relevantJoints,newXRefs):
                self.jointTargetPos[jointName] = angle
            
            positionsBeforeTick = {jointName:jointPos for jointName,jointPos in zip(relevantJoints, self.getJointAngles(relevantJoints)) }
            
            self.tick(jointPrevPositions)
            jointPrevPositions = positionsBeforeTick
        
        currEndEffPos = self.getJointPosition(endEffector)

        while self.getVectorLength(targetPosition - currEndEffPos.squeeze()) > threshold:
            positionsBeforeTick = {jointName:jointPos for jointName,jointPos in zip(relevantJoints, self.getJointAngles(relevantJoints)) }

            self.tick(jointPrevPositions)
            jointPrevPositions = positionsBeforeTick
            
            currEndEffPos = self.getJointPosition(endEffector)

        return 


    def tick(self, jointPrevPositions=None):
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
            jointPosition = self.getJointPos(joint)
            if jointPrevPositions:
                currVelocity = (jointPosition - jointPrevPositions.get(joint, jointPosition))/ self.dt
            else:
                currVelocity = 0
            torque = self.calculateTorque(self.jointTargetPos[joint],jointPosition, 0.0, currVelocity, 0, kp, ki, kd)
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

    # Accepts an Nx3 array of data points
    def cubic_interpolation(self, points, nTimes=100):
        """
        Given a set of control points, return the
        cubic spline defined by the control points,
        sampled nTimes along the curve.
        """
        
        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]
        inputTimes = np.arange(0,points.shape[0])
        
        xSpline = CubicSpline(inputTimes, xs)
        ySpline = CubicSpline(inputTimes, ys)
        zSpline = CubicSpline(inputTimes, zs)

        outputTimes = np.linspace(inputTimes[0], inputTimes[-1], nTimes)

        return xSpline(outputTimes), ySpline(outputTimes), zSpline(outputTimes)

    # Task 3.1 Pushing
    def dockingToPosition(self, leftTargetAngle, rightTargetAngle, angularSpeed=0.005,
            threshold=1e-1, maxIter=300, verbose=False):
        """A template function for you, you are free to use anything else"""
        pass

    # Task 3.2 Grasping & Docking
    def clamp(self, leftTargetAngle, rightTargetAngle, angularSpeed=0.005, threshold=1e-1, maxIter=300, verbose=False):
        """A template function for you, you are free to use anything else"""
        pass
                        
    # Moves both of the robot's end effectors at the same time with PD. First computes the IK configurations for each, then loops and ticks while updating both in tandem
    # Disables the shared chest joint to prevent conflicting instructions from the IK solvers
    def moveBothHands(self, leftTargetPos, leftTargetOrientation, rightTargetPos, rightTargetOrientation, speed=0.01, maxIter=3000):
        
        leftEndEffector = 'LHAND'
        rightEndEffector = 'RHAND'
        
        leftRelevantJoints = self.getRelevantJoints(leftEndEffector)
        leftRelevantJoints.remove('CHEST_JOINT0')

        rightRelevantJoints = self.getRelevantJoints(rightEndEffector)
        rightRelevantJoints.remove('CHEST_JOINT0')

        bothRelevantJointSets = leftRelevantJoints + rightRelevantJoints

        left_start_pos = self.getJointPosition(leftEndEffector).squeeze()
        right_start_pos = self.getJointPosition(rightEndEffector).squeeze()

        
        leftNumSteps = self.calIterToTarget(left_start_pos,leftTargetPos, speed)
        rightNumSteps = self.calIterToTarget(right_start_pos,leftTargetPos, speed)
        numSteps = max(leftNumSteps, rightNumSteps)
        numSteps = min(numSteps, maxIter)

        leftJointRefs = self.inverseKinematics(leftEndEffector, leftTargetPos, leftTargetOrientation, {"steps": numSteps, "relevantJoints": leftRelevantJoints})
        rightJointRefs = self.inverseKinematics(rightEndEffector, rightTargetPos, rightTargetOrientation, {"steps": numSteps, "relevantJoints": rightRelevantJoints})
        jointPrevPositions = {jointName:jointPos for jointName,jointPos in zip(bothRelevantJointSets, self.getJointAngles(bothRelevantJointSets)) }

        for i in range(leftJointRefs.shape[1]):

            leftJointConfigs = leftJointRefs[:,i]
            rightJointConfigs = rightJointRefs[:,i]

            bothSetsOfJointConfigs = np.hstack((leftJointConfigs, rightJointConfigs))
            for jointName,angle in zip(bothRelevantJointSets,bothSetsOfJointConfigs):
                self.jointTargetPos[jointName] = angle

            positionsBeforeTick = {jointName:jointPos for jointName,jointPos in zip(bothRelevantJointSets, self.getJointAngles(bothRelevantJointSets)) }
            
            self.tick(jointPrevPositions)
            jointPrevPositions = positionsBeforeTick  
        
        return

    # Moves a target joint while sending torques to other joints in order for them to maintain their configurations. Calls the actual tick()
    def moveJointInConjuction(self, jointName, targetPosition, angularSpeed=0.1, maxIter=3000):
        startPosition = self.getJointPos(jointName)

        numSteps = min(maxIter, self.calIterToTarget(startPosition,targetPosition, angularSpeed))
        print(numSteps)
        jointPositions = np.linspace(startPosition, targetPosition, numSteps)

        otherJoints = self.joints
        if jointName != 'CHEST_JOINT0':
            otherJoints.remove(jointName)

        for joint in otherJoints:
            self.jointTargetPos[joint] = self.getJointPos(joint)

        jointPrevPositions = {jointName:jointPos for jointName,jointPos in zip(self.joints, self.getJointAngles(self.joints)) }

        for pos in jointPositions:
            self.jointTargetPos[jointName] = pos

            positionsBeforeTick = {jointName:jointPos for jointName,jointPos in zip(self.joints, self.getJointAngles(self.joints)) }

            self.tick(jointPrevPositions)
            jointPrevPositions = positionsBeforeTick

        return
    
    # "On the fly" implementation of the moveBothHands() function which calls single step inverse kinematics at each iteration rather than calculating in advance
    def moveBothHandsOnTheFly(self,leftTargetPos, leftTargetOrientation, rightTargetPos, rightTargetOrientation, speed=0.01, maxIter=3000):
        leftEndEffector = 'LHAND'
        rightEndEffector = 'RHAND'
        
        usingLeftOrientation =  np.any(leftTargetOrientation)
        usingRightOrientation = np.any(rightTargetOrientation)

        leftRelevantJoints = self.getRelevantJoints(leftEndEffector)
        leftRelevantJoints.remove('CHEST_JOINT0')
        rightRelevantJoints = self.getRelevantJoints(rightEndEffector)
        rightRelevantJoints.remove('CHEST_JOINT0')
        bothRelevantJointSets = leftRelevantJoints + rightRelevantJoints

        left_start_pos = self.getJointPosition(leftEndEffector).squeeze()
        right_start_pos = self.getJointPosition(rightEndEffector).squeeze()
        
        leftNumSteps = self.calIterToTarget(left_start_pos,leftTargetPos, speed)
        rightNumSteps = self.calIterToTarget(right_start_pos,leftTargetPos, speed)
        numSteps = max(leftNumSteps, rightNumSteps)
        numSteps = min(numSteps, maxIter)

        leftPositionTrajectory = np.linspace(left_start_pos, leftTargetPos, numSteps)
        rightPositionTrajectory = np.linspace(right_start_pos, rightTargetPos, numSteps)

        if usingLeftOrientation:
            leftStartOrientation = self.getJointOrientation(leftEndEffector)
            leftOrientationTrajectory = np.linspace(leftStartOrientation, leftTargetOrientation, numSteps)

        if usingRightOrientation:
            rightStartOrientation = self.getJointOrientation(rightEndEffector)
            rightOrientationTrajectory = np.linspace(rightStartOrientation, rightTargetOrientation, numSteps)

        jointPrevPositions = {jointName:jointPos for jointName,jointPos in zip(bothRelevantJointSets, self.getJointAngles(bothRelevantJointSets)) }

        for i in range(1,numSteps):
            leftNextPosition = leftPositionTrajectory[i]
            rightNextPosition = rightPositionTrajectory[i]

            leftNextOrientation = None
            rightNextOrientation = None
            if usingLeftOrientation:
                leftNextOrientation = leftOrientationTrajectory[i]

            if usingRightOrientation:
                rightNextOrientation = rightOrientationTrajectory[i]

            leftJointConfigs = self.singleStepInverseKinematics(leftEndEffector, leftNextPosition, leftNextOrientation, {"relevantJoints": leftRelevantJoints})
            rightJointConfigs = self.singleStepInverseKinematics(rightEndEffector, rightNextPosition, rightNextOrientation, {"relevantJoints": rightRelevantJoints})

            bothSetsOfJointConfigs = np.hstack((leftJointConfigs, rightJointConfigs))
            for jointName,angle in zip(bothRelevantJointSets,bothSetsOfJointConfigs):
                self.jointTargetPos[jointName] = angle
            
            positionsBeforeTick = {jointName:jointPos for jointName,jointPos in zip(bothRelevantJointSets, self.getJointAngles(bothRelevantJointSets)) }
            
            self.tick(jointPrevPositions)
            jointPrevPositions = positionsBeforeTick
        
        return
 ### END
