import threading
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rclpy
from rclpy.node import Node
from rclpy.exceptions import ROSInterruptException
import signal
import numpy as np
from group_project import coordinates
from group_project import AJBastroalign
import matplotlib.pyplot as plt
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import Twist,Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image,LaserScan,CameraInfo
from math import sin, cos
import time
import math
import random
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from group_project.distance_between import Distance_Between
from group_project.detect_circle import Detect_Circle
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Odometry

class RoboNaut(Node):
    def __init__(self):
        super().__init__('robotnaut')

        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Initialise variables
        self.position = None
        self.orientation = None
        
        # Initialise flags
        self.goal_achieved = False
        self.entrance_reached = False
        self.entrance_1_reached = False
        self.entrance_2_reached = False
        
        self.rotate_robot = False
        self.pause_robot = False
        
        self.green_circle_detected = False
        self.red_circle_detected = False
        self.green_detected = False
        self.red_detected = False
        self.sensitivity = 50
        self.green_room_reached = False  
        self.robot_x = None
        self.robot_y = None
        self.start_window_detection = None
        self.regions = {'front': 0}
        self.last_turn_direction = None
        self.safe_to_go_forward = 2.0
        self.distance_threshold = 0.001
        self.goal_x = 0.0 
        self.goal_y = 0.0 
        self.camera_width = None
        self.camera_height = None
        self.window_center_x = None
        self.window_center_y = None
        self.window_detected = False 
        self.cv_image_width = None
        self.cv_image_height = None
        self.cv_image = None 
             
               
        
        self.declare_parameter('coordinates_file_path', '')
        coordinates_file_path = self.get_parameter('coordinates_file_path').get_parameter_value().string_value
        self.coordinates = coordinates.get_module_coordinates(coordinates_file_path)
        self.window_detected = False
        self.should_continue = True
        self.captured_windows = []
        self.captured_images = []  # Keep track of the file names of captured images

        # Initialize the CvBridge and subscribe to the camera image topic
        self.bridge = CvBridge()
        self.window_counter = 0  # Keep track of the number of windows detected
        # self.subscription = self.create_subscription(Image,'/camera/image_raw', self.window_callback,10)
    
        
        # Subscription for the camera input from the robot
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.subscription  # prevent unused variable warning
        
        self.subscription = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        
        # Publisher for movement
        self.movePublisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.subscription = self.create_subscription(Pose, '/robot_pose', self.pose_callback, 10)   
        
        
    
        self.rate = self.create_rate(10)  # 10 Hz
    
        # You can access the module coordinates like so:
        # Room 1:
        self.coordinates.module_1.entrance.x
        self.coordinates.module_1.entrance.y
        self.coordinates.module_1.center.x
        self.coordinates.module_1.center.y

        # Room 2:
        self.coordinates.module_2.entrance.x
        self.coordinates.module_2.entrance.y
        self.coordinates.module_2.center.x
        self.coordinates.module_2.center.y

        self.goal_publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)
        # Subscriber for odometry
        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        # Current robot pose
        self.current_pose = None

    # Function to calculate the distance between the Earth and the Moon from the image
    def calculate_distance(self):
        
        earth_diameter_km = 12742  
        scaling_factor = 3  

        # Load the image
        image = cv2.imread('../ros2_ws/Panorama.jpg')
        if image is None:
            raise ValueError(f"Image at path could not be loaded. Please check the file path and try again.")

        # Process the image to find contours of Earth and Moon
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        radii = []
        centers = []
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            centers.append((int(x), int(y)))
            radii.append(radius / scaling_factor)  # Apply scaling factor here

        # Calculate distances
        if len(centers) == 2:  # Ensure two centers were found
            center_distance_pixels = np.sqrt((centers[0][0] - centers[1][0])**2 + (centers[0][1] - centers[1][1])**2)
            if center_distance_pixels > sum(radii):
                circumference_distance_pixels = center_distance_pixels - sum(radii)
                scale_km_per_pixel = earth_diameter_km / (2 * radii[0] * scaling_factor)  # Adjusted scale conversion
                real_world_distance_km = circumference_distance_pixels * scale_km_per_pixel
            else:
                real_world_distance_km = 0  # Indicates overlap or touching
        else:
            real_world_distance_km = -1  # Error indicator
            
        print(f"Real-world distance: {real_world_distance_km} km")
        return real_world_distance_km
        
    def send_goal(self, x, y, yaw):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Position
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y

        # Orientation
        goal_msg.pose.pose.orientation.z = sin(yaw / 2)
        goal_msg.pose.pose.orientation.w = cos(yaw / 2)

        #Get the goal position
        self.goal_x = x
        self.goal_y = y
        
        self.action_client.wait_for_server()
        self.send_goal_future = self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        try:
            result = future.result()
            self.goal_achieved = True 
            
            if result.status == GoalStatus.STATUS_SUCCEEDED and not self.pause_robot:
                #return true when the green room is reached 
                self.green_room_reached = True
            
        except Exception as e:
            self.goal_achieved = False

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
 
                     
    def image_callback(self, msg):
        
        self.green_circle_detected, self.red_circle_detected = Detect_Circle(msg, self.sensitivity)
     
        if self.green_room_reached:  # Only detect windows if it is in the green room
            windows = self.find_windows(cv_image)
            for window in windows:
                window_position = self.calculate_window_global_position(window)
                self.send_goal_to_nav2(window_position[0], window_position[1])
    
            if not self.window_detected:
                
                self.window_detected = self.detect_window()
                if self.window_detected:
                    self.align_with_window()

                
                    
        bridge = CvBridge()
        try:
            # Convert from ROS Image message to OpenCV image
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.cv_image_width = cv_image.shape[1]
            self.cv_image_height = cv_image.shape[0]
            self.cv_image = cv_image            
            cv2.imshow("Camera View", self.cv_image)
            cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)
    
    def find_windows(self, image):
        # Assuming image is a grayscale image where windows are detectable by edges
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        windows = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 30 and h > 30:  # Example criteria
                windows.append((x + w // 2, y + h // 2))
        return windows
    
    
    def align_with_window(self):
        camera_center_x = self.cv_image_width / 2
        window_center_x = self.window_center_x 
        error_x = window_center_x - camera_center_x
        self.integral = 0
        self.last_error = 0
    
        # PID parameters
        Kp = 0.01  # Proportional gain
        Ki = 0.001  # Integral gain
        Kd = 0.005  # Derivative gain

        # Proportional control
        P = Kp * error_x

        # Integral control
        self.integral += error_x
        I = Ki * self.integral

        # Derivative control
        D = Kd * (error_x - self.last_error)
        self.last_error = error_x

        # Combine for PID control
        angular_velocity = -(P + I + D)

        msg = Twist()
        msg.angular.z = angular_velocity
        self.movePublisher.publish(msg)

    def send_goal_to_nav2(self, window_x, window_y):
        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = "map"
        goal_pose.pose.position.x = window_x
        goal_pose.pose.position.y = window_y
        goal_pose.pose.orientation.w = 1.0
        self.goal_publisher.publish(goal_pose)
        self.get_logger().info(f"Goal sent to Nav2: Position ({window_x}, {window_y})")

    def odom_callback(self, msg):
        orientation_q = msg.pose.pose.orientation
        yaw_angle = self.quaternion_to_euler(orientation_q)
        self.current_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw_angle)

    def quaternion_to_euler(self, quaternion):
        x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw
    
    def calculate_window_global_position(self, window_pixel):
        """
        Calculate the window's global position based on its pixel coordinates in the image.
        
        Args:
        window_pixel (tuple): Pixel coordinates (x, y) of the window in the image.

        Returns:
        tuple: Global coordinates (x, y) of the window in the map.
        """
        # Placeholder values for camera parameters and robot's pose
        focal_length = 1.0  # Example focal length, needs calibration data
        robot_pose = self.current_pose  # Assuming this contains (x, y, theta) from odometry or localization

        # Convert pixel coordinates to angles
        image_center_x = self.cv_image_width / 2
        image_center_y = self.cv_image_height / 2
        angle_x = math.atan2((window_pixel[0] - image_center_x), focal_length)
        angle_y = math.atan2((window_pixel[1] - image_center_y), focal_length)

        # Assuming the depth to the window is known or estimated
        depth = self.estimate_depth(window_pixel)  # You need to implement this method based on your setup

        # Calculate window position in robot's coordinate frame
        local_x = depth * math.sin(angle_x)
        local_y = depth * math.sin(angle_y)

        # Transform to global coordinates
        global_x = robot_pose[0] + local_x * math.cos(robot_pose[2]) - local_y * math.sin(robot_pose[2])
        global_y = robot_pose[1] + local_x * math.sin(robot_pose[2]) + local_y * math.cos(robot_pose[2])

        return (global_x, global_y)
            
        
    def detect_window(self):
        
        if not self.window_detected and self.green_room_reached:  
            try:
            #    cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    epsilon = 0.05 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) == 4:  # Rectangle check
                        x, y, w, h = cv2.boundingRect(approx)
                        if w > 30 and h > 30:  # Size filter
                            window_candidate = self.cv_image[y:y+h, x:x+w]
                            
                            # Define border width
                            border_width = 10  # Adjust as needed
                            
                            # Create a mask for the border region
                            border_mask = np.zeros_like(window_candidate[:, :, 0], dtype=bool)
                            border_mask[:border_width, :] = True  # Top border
                            border_mask[-border_width:, :] = True  # Bottom border
                            border_mask[:, :border_width] = True  # Left border
                            border_mask[:, -border_width:] = True  # Right border
                            
                            # Check if the border regions are white
                            border_threshold = 220  # Adjust based on your lighting conditions
                            is_white_border = np.all(window_candidate[border_mask, :] > border_threshold, axis=1)
                            
                            # Calculate the percentage of white border pixels
                            white_border_ratio = np.sum(is_white_border) / is_white_border.size
                            
                            if white_border_ratio > 0.1:  # Threshold to adjust based on your observations
                                
                                
                               
                               # self.get_logger().info("Window detected based on white border.")
                                
                                #Call the spatial transformation to make the robot move towards a window 
                                self.window_center_x = x + w / 2
                                self.window_center_y = y + h / 2
                               
                               # Draw rectangle on the image
                                top_left_corner_x = int(self.window_center_x - w / 2)
                                top_left_corner_y = int(self.window_center_y - h / 2)
                                bottom_right_corner_x = int(self.window_center_x + w / 2)
                                bottom_right_corner_y = int(self.window_center_y + h / 2)
                                cv2.rectangle(self.cv_image, (top_left_corner_x, top_left_corner_y), (bottom_right_corner_x, bottom_right_corner_y), (0, 255, 0), 2)
                               
                                self.stop()
                                self.get_logger().info("Window located")
                                 
                                # window_filename = f'window{self.window_counter}.png'
                                # cv2.imwrite(window_filename, window_candidate)
                                # self.get_logger().info(f'Image saved as {window_filename}')
                                return True
                                # self.should_continue = True 
                                # self.captured_images.append(window_filename)
                                # if self.window_counter == 2:
                                #     self.get_logger().info("Starting image stitching process.")
                                #     self.stitch_images()
                            
            except CvBridgeError as e:
                self.get_logger().error(f'Failed to convert image: {e}')
        
        
        

    def laser_callback(self,msg):
        #Laser scan has inf values which are valid pth therefore replace inf with 0 
        # We limit the robot frontal sensor        
        front_ranges = msg.ranges[0:20] + msg.ranges[-20:]
        self.regions = {
'front': min(front_ranges) if front_ranges else 0
        } 
        
    def pose_callback(self,msg):
        self.robot_pose = msg
    
    #Code for Moving around the room and detecting the screenshots
    #We have to use lidar sensor to move around the room
    #Random Walking until a certain threshold is met 
    def move_robot(self):
        
            if not self.window_detected:
                msg = Twist()
                #  threshold for obstacles
                if self.regions['front'] > self.safe_to_go_forward: 
                # self.get_logger().info("Moving robot")
                #   print("Safe to move forwad")
                    msg.linear.x = 0.4 # move forward at 0.5 m/s
                    self.last_turn_direction = None
                else:
                    self.stop()
                    # choose randomly to either move left or right
                    if self.last_turn_direction is None:
                        self.last_turn_direction = random.choice([-1, 1]) * 0.5 
            
                    msg.angular.z =  self.last_turn_direction 

                self.movePublisher.publish(msg)
            
    def rotate_360(self):
        # Initialize unused components of desired velocity to zero
        desired_velocity = Twist()
        # Set desired anglular velocity
        desired_velocity.angular.z = 0.2
        # store current time: t0
        t0, _ = self.get_clock().now().seconds_nanoseconds()
        current_angle = 0
        # loop to publish the velocity estimate until desired angle achieved
        # current angle = current angular velocity * (t1 - t0)
        while (current_angle < 2*math.pi):
            if (self.green_circle_detected or self.red_circle_detected):
                break
            else:
                # Publish the velocity
                self.movePublisher.publish(desired_velocity)
                # t1 is the current time
                t1, _ = self.get_clock().now().seconds_nanoseconds()  # to_msg()
                # Calculate current angle
                current_angle = desired_velocity.angular.z * (t1 - t0)
                self.rate.sleep()
        # set velocity to zero to stop the robot
        self.stop()

    def stop(self):
        desired_velocity = Twist()  # Creates a new Twist message
        desired_velocity.linear.x = 0.0  
        desired_velocity.linear.y = 0.0 
        desired_velocity.linear.z = 0.0
        desired_velocity.angular.x = 0.0
        desired_velocity.angular.y = 0.0  
        desired_velocity.angular.z = 0.0
        self.movePublisher.publish(desired_velocity)

    def pause(self):
        for _ in range(10):
            self.rate.sleep()
        
def main():
    def signal_handler(sig, frame):
        print("Shutting down...")
        robonaut.should_continue = False
        robonaut.stop()
        rclpy.shutdown()
        
    rclpy.init(args=None)
    robonaut = RoboNaut()
    signal.signal(signal.SIGINT, signal_handler)


    thread = threading.Thread(target=rclpy.spin, args=(robonaut,), daemon=True)
    thread.start()
  
    while rclpy.ok():
        
      
  #      Find the nearest entrance and go to it
        if (robonaut.position != None):
            if (robonaut.goal_achieved == False):
                if robonaut.pause_robot == True:
                    robonaut.pause()
                else:
                    if (Distance_Between(robonaut.position, robonaut.coordinates.module_1.entrance) > Distance_Between(robonaut.position, robonaut.coordinates.module_2.entrance)):
                        robonaut.send_goal(robonaut.coordinates.module_2.entrance.x, robonaut.coordinates.module_2.entrance.y,0.0)               
                        robonaut.entrance_2_reached = True
                        robonaut.pause_robot = True
                    else:
                        robonaut.send_goal(robonaut.coordinates.module_1.entrance.x, robonaut.coordinates.module_1.entrance.y,0.0)
                        robonaut.entrance_1_reached = True
                        robonaut.pause_robot = True

            elif (robonaut.goal_achieved == True):
                robonaut.pause_robot = False
                if robonaut.rotate_robot == False:
                    robonaut.rotate_360()
                    robonaut.rotate_robot = True
                if robonaut.green_circle_detected:
                    if robonaut.pause_robot == False:
                        if(robonaut.entrance_1_reached):
                            robonaut.send_goal(robonaut.coordinates.module_1.center.x, robonaut.coordinates.module_1.center.y,0.0)
                        else:
                            robonaut.send_goal(robonaut.coordinates.module_2.center.x, robonaut.coordinates.module_2.center.y,0.0)
                    else:
                        robonaut.pause()
                elif robonaut.red_circle_detected:
                    if robonaut.pause_robot == False:
                        if(robonaut.entrance_1_reached):
                            robonaut.send_goal(robonaut.coordinates.module_2.center.x, robonaut.coordinates.module_2.center.y,0.0)
                        else:
                            robonaut.send_goal(robonaut.coordinates.module_1.center.x, robonaut.coordinates.module_1.center.y,0.0)
                    else:
                       robonaut.pause()
               
               
                # Reach the green room and start random movement
                if robonaut.green_room_reached:    
                    robonaut.move_robot()
                    if robonaut.window_detected:   
                        pass
                        #robonaut.stop()

                        
           
                
        # robonaut.move_robot()

if __name__ == "__main__":
    main()