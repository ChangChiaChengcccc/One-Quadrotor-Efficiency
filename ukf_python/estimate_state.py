#!/usr/bin/env python

import rospy
from ukf import UKF
import numpy as np
import math 
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from pyquaternion import Quaternion
from mav_msgs.msg import Actuators
from sensor_msgs.msg import Imu
from geometry_msgs.msg import WrenchStamped

# time variables
time_last = 0
dt = 0

# pre-declare variables
state_dim = 6
measurement_dim = 3
sensor_data = np.zeros(measurement_dim)
Rot_mat = np.zeros((3,3))
allo_mat = np.array([
                     [1,   1,   1,   1],
                     [-0.255539*math.sin(1.0371),0.238537*math.sin(0.99439),0.255539*math.sin(1.0371),-0.238537*math.sin(0.99439)],
                     [-0.255539*math.cos(1.0371),0.238537*math.cos(0.99439),-0.255539*math.cos(1.0371),0.238537*math.cos(0.99439)],
                     [-1.6e-2,   -1.6e-2,   1.6e-2,   1.6e-2]    
                     ])
f_vector = np.zeros(4)
f_vector_sensor = np.zeros(4)
f_M = np.zeros(4)
f_M_sensor = np.zeros(4)
acc = np.zeros(3)
acc_imu = np.zeros(3)
acc_dyn = np.zeros(3)
e3 = np.array([0,0,1])
f0 = 0
f1 = 0
f2 = 0
f3 = 0

estimate_state_list = Float64MultiArray()
acc_list = Float64MultiArray()
debug = np.zeros(3)
debug_list = Float64MultiArray()


# Process Noise
q = np.eye(state_dim)
# x,v
q[0][0] = 0.0001 
q[1][1] = 0.0001
q[2][2] = 0.0001
q[3][3] = 0.05
q[4][4] = 0.05
q[5][5] = 0.05

# create measurement noise covariance matrices
p_yy_noise = np.eye(measurement_dim)
p_yy_noise[0][0] = 0.0001
p_yy_noise[1][1] = 0.0001
p_yy_noise[2][2] = 0.0001

# create initial state
initial_state = np.zeros(state_dim)


def iterate_x(x, timestep):
    '''this function is based on the x_dot and can be nonlinear as needed'''
    global acc, acc_imu, acc_dyn, f_M, e3, debug, f_M_sensor, f0,f1,f2,f3, f_vector_sensor
    # dynamics
    f_vector_sensor = np.array([f0,f1,f2,f3])
    f_M_sensor = np.dot(allo_mat,f_vector_sensor)
    acc = np.dot(Rot_mat,acc_imu) - 9.8*e3
    acc_dyn = f_M[0]*np.dot(Rot_mat,e3)/1.52 - 9.8*e3
    debug = acc_dyn


    ret = np.zeros(len(x))
    # x,v
    ret[0] = x[0] + x[3] * timestep
    ret[1] = x[1] + x[4] * timestep
    ret[2] = x[2] + x[5] * timestep
    ret[3] = x[3] 
    ret[4] = x[4] 
    ret[5] = x[5] 

    return ret

def measurement_model(x):
    """
    :param x: states
    """
    # dynamics

    global measurement_dim
    ret = np.zeros(measurement_dim)
    ret[0] = x[0]
    ret[1] = x[1]
    ret[2] = x[2]
    return ret

def odometry_cb(data):
    global Rot_mat
    global sensor_data
    quaternion = Quaternion(data.pose.pose.orientation.w, data.pose.pose.orientation.x, 
                            data.pose.pose.orientation.y, data.pose.pose.orientation.z)
    Rot_mat = quaternion.rotation_matrix
    sensor_data =  np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z])

def imu_cb(data):
    global acc_imu
    acc_imu = np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z])

def rotors_cb(data):
    global f_vector,allo_mat,f_M
    rotor_force_constant = 8.44858e-06 #8.54858e-06
    f_vector = np.array([data.angular_velocities[0]*data.angular_velocities[0]*rotor_force_constant,
                         data.angular_velocities[1]*data.angular_velocities[1]*rotor_force_constant,
                         data.angular_velocities[2]*data.angular_velocities[2]*rotor_force_constant, 
                         data.angular_velocities[3]*data.angular_velocities[3]*rotor_force_constant])

    f_M = np.dot(allo_mat,f_vector)
    # print("f_vector:")
    # print(f_vector)
    # print("f_M:")
    # print(f_M)

def ft0_cb(data):
    global f0
    f0 = data.wrench.force.z

def ft1_cb(data):
    global f1
    f1 = data.wrench.force.z

def ft2_cb(data):
    global f2
    f2 = data.wrench.force.z

def ft3_cb(data):
    global f3
    f3 = data.wrench.force.z

def ukf():
    global time_last
    global dt
    dt = rospy.Time.now().to_sec() - time_last
    ukf_module.predict(dt)
    ukf_module.update(measurement_dim, sensor_data, p_yy_noise)
    time_last = rospy.Time.now().to_sec()

    # print('dt:')
    # print(dt)
    # print('rospy.Time.now().to_sec()')
    # print(rospy.Time.now().to_sec())


if __name__ == "__main__":
    try:
        rospy.init_node('UKF')
        state_pub = rospy.Publisher("/estimated_state", Float64MultiArray, queue_size=10)
        acc_pub = rospy.Publisher("/iris1/acc", Float64MultiArray, queue_size=10)
        debug_pub = rospy.Publisher("/debug", Float64MultiArray, queue_size=10)
        rospy.Subscriber("/iris1/ground_truth/odometry", Odometry, odometry_cb, queue_size=10)
        rospy.Subscriber("/iris1/motor_speed", Actuators, rotors_cb, queue_size=10)
        rospy.Subscriber("/iris1/ground_truth/imu", Imu, imu_cb, queue_size=10)
        rospy.Subscriber("/iris1/iris1/rotor_0_ft", WrenchStamped, ft0_cb, queue_size=10)
        rospy.Subscriber("/iris1/iris1/rotor_1_ft", WrenchStamped, ft1_cb, queue_size=10)
        rospy.Subscriber("/iris1/iris1/rotor_2_ft", WrenchStamped, ft2_cb, queue_size=10)
        rospy.Subscriber("/iris1/iris1/rotor_3_ft", WrenchStamped, ft3_cb, queue_size=10)
        # pass all the parameters into the UKF!
        # number of state variables, process noise, initial state, initial coariance, three tuning paramters, and the iterate function
        #def __init__(self, num_states, process_noise, initial_state, initial_covar, alpha, k, beta, iterate_function, measurement_model):
        ukf_module = UKF(state_dim, q, initial_state, 0.001*np.eye(state_dim), 0.001, 0.0, 2.0, iterate_x, measurement_model)
        rate = rospy.Rate(40)
        while not rospy.is_shutdown():         
            ukf()
            estimate_state = ukf_module.get_state()
            estimate_state_list.data = list(estimate_state)
            state_pub.publish(estimate_state_list)

            acc_list.data = list(acc)
            acc_pub.publish(acc_list)

            debug_list.data = list(debug)
            debug_pub.publish(debug_list)

            rate.sleep()
    except rospy.ROSInterruptException:
        pass