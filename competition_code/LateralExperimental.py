import numpy as np
import math
import roar_py_interface

def normalize_rad(rad: float):
    #range [0, 2*pi]
    return rad % (2 * np.pi)

class LatController:
    def run(self, vehicle_location, vehicle_rotation, waypoints, current_section) -> float:
        """
        Calculates the steering command using the pure pursuit algorithm.
        Adjusted to consider an averaged waypoint for smoother turns.
        Args:
            vehicle_location (np.ndarray): Current vehicle [x, y, z] location.
            vehicle_rotation (np.ndarray): Current vehicle rotation [roll, pitch, yaw].
            waypoints (list or RoarPyWaypoint): A list of waypoints or a single waypoint.
        Returns:
            float: Steering command.
        """
        # make sure waypoints are list
        if isinstance(waypoints, roar_py_interface.RoarPyWaypoint):
            waypoints = [waypoints]

        if len(waypoints) < 1:
            return 0

        x_const = 0.5
        # 20 waypoints
        num_waypoints_to_consider = min(20, len(waypoints))
        weights = np.zeros(num_waypoints_to_consider)
        for i in range(num_waypoints_to_consider):
            weights[i] = ((1+ (x_const*(i)/(num_waypoints_to_consider)) )/np.linalg.norm(vehicle_location[:2] - waypoints[i].location[:2]))

        weight_sum = 0;
        for i in range(num_waypoints_to_consider):
            weight_sum += weights[i]*waypoints[i].location[:2]
        weight_sum /= sum(weights)

        # waypoint_weights = np.linspace(0.2, 1.5, num=num_waypoints_to_consider) #originally 1-1.5

        # #Value each waypoint based on its weight and the distance to the vehicle
        # weighted_location = sum(
        #     w.location[:2] * (waypoint_weights[i] / np.linalg.norm(vehicle_location[:2] - w.location[:2]))
        #     for i, w in enumerate(waypoints[:num_waypoints_to_consider])
        # ) / sum(
        #     waypoint_weights[i] / np.linalg.norm(vehicle_location[:2] - w.location[:2]) for i, w in enumerate(waypoints[:num_waypoints_to_consider])
        # )

        # temporary averaged waypoint
        avg_waypoint = roar_py_interface.RoarPyWaypoint(
            location=np.array([*weight_sum, 0]),  # Add z=0
            roll_pitch_yaw=[0, 0, 0],
            lane_width=0
        )

        # Calculate steering based on the averaged waypoint
        return self.calculate_steering(vehicle_location[:2], vehicle_rotation, avg_waypoint, current_section)

    def calculate_steering(self, vehicle_location, vehicle_rotation, waypoint, current_section) -> float:
        """
        Calculates the steering command using the pure pursuit algorithm for a single waypoint
        Args:
            vehicle_location (tuple): Current [x, y] location of the vehicle.
            vehicle_rotation (np.ndarray): Current vehicle rotation [roll, pitch, yaw].
            waypoint (RoarPyWaypoint): The target waypoint to calculate steering for.
        Returns:
            float: Steering command.
        """
        #x, y coordinates of the waypoint
        waypoint_x, waypoint_y = waypoint.location[:2]

        # Calculate vector pointing from vehicle to the waypoint
        waypoint_vector = np.array([waypoint_x, waypoint_y]) - np.array(vehicle_location)

        # distance to the waypoint
        distance_to_waypoint = np.linalg.norm(waypoint_vector)
        if distance_to_waypoint == 0:
            return 0  

        waypoint_vector_normalized = waypoint_vector / distance_to_waypoint

        # Angle to waypoint 
        alpha = normalize_rad(vehicle_rotation[2]) - normalize_rad(
            math.atan2(waypoint_vector_normalized[1], waypoint_vector_normalized[0])
        )

        # Pure pursuit formula
        steer_multiplier = 1.50

        # if current_section == 1:
        #     steer_multiplier = 1.35
        # if current_section == 2:
        #     steer_multpilier = 1.35
        # if current_section == 4:
        #     steer_multpilier = 1.20
        # if current_section == 5:
        #     steer_multpilier = 1.20
        # if current_section == 6:
        #     steer_multpilier = 1.20

        steering_command = steer_multiplier * math.atan2(
            2.0 * 4.7 * math.sin(alpha) / distance_to_waypoint, 1.0
        )

        return float(steering_command)