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
        assert "relevantJoints" in frame

        relevantJoints = frame.get('relevantJoints')
        start_pos, start_orientation = self.getJointLocationAndOrientation(endEffector)

        xRefDeltas = np.zeros(len(relevantJoints))
        # print(currEndEffPos)
        currJacobian = self.jacobianMatrix(endEffector, relevantJoints)

        if np.any(orientation):
            xRefDeltas = np.linalg.pinv(currJacobian) @ ((np.hstack((targetPosition, orientation))) - np.hstack((start_pos,start_orientation @ self.refVector)))
        else:
            xRefDeltas = np.linalg.pinv(currJacobian[:3]) @ np.squeeze(targetPosition - start_pos)
        xRefs = np.array(self.getJointAngles(relevantJoints)) + xRefDeltas

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

        start_pos = self.getJointPosition(endEffector)

        # print(targetPosition)
        # print(start_pos)
        numSteps = int(np.linalg.norm(targetPosition-start_pos)//speed)
        numSteps = min(numSteps, maxIter)
        # print(numSteps)

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
            
            newXRefs = self.inverseKinematics(endEffector, nextPosition, nextOrientation, {"relevantJoints": relevantJoints})

            for jointName,angle in zip(relevantJoints,newXRefs):
                self.p.resetJointState(
                    self.robot, self.jointIds[jointName], angle)

            pltTime.append(i)
            pltDistance.append(np.linalg.norm(nextPosition - start_pos))
            self.tick_without_PD(relevantJoints, newXRefs)

        return pltTime, pltDistance

    def tick_without_PD(self, relevantJoints=None, jointPositions=None):
        """Ticks one step of simulation without PD control. """
        # TODO modify from here
        # Iterate through all joints and update joint states.
            # For each joint, you can use the shared variable self.jointTargetPos.

        assert relevantJoints is not None
        assert jointPositions is not None
        
        for jointName,angle in zip(relevantJoints,jointPositions):
                self.jointTargetPos[jointName] = angle

        self.p.stepSimulation()
        self.drawDebugLines()
        time.sleep(self.dt)