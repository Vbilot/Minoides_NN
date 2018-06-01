# online_trainer.py

# The OnlineTrainer class monitor a lesson until an ending criterion is met
# There are 3 ending criterion :
#     - Interruption by the run.py script (self.running set to False)
#     - Convergence of the robot : the relative error must stay under a given threshold for a given duration
#     - Maximum experiment duration reached
# These parameter can be set in the config.cfg file

# The trainer is run on a new thread to enable run.py to stop it
# It performs several actions :
#     - Calling the Neuron Network iterations
#     - Setting the robot velocity
#     - Computing the gradient
#     - Calling the backpropagation

import time
import math
import json
from math import cos, sin, atan2, exp, log, pi, sqrt, atan
import os
from movement_equations import get_grad

'''
def porte_theta(x):
    if x<-pi/2 or x>pi/2:
        return pi
    if x>-pi/2 and x<0:
        return (-2)*x
    else:
        return (2)*x


def porte_theta(x):
    if x>0:
        return exp(x/pi)-1
    else:
        return exp(-x/pi)-1

'''
def create_folder_if_necessary(dirname):
    if "/" in dirname:
        l = os.listdir(dirname[:dirname.rfind("/")])
    else:
        l = os.listdir()
    if dirname[dirname.find("/")+1:] not in l:
        os.mkdir(dirname)
    elif not os.path.isdir(dirname):
        os.remove(dirname)
        os.mkdir(dirname)

def create_path_if_necessary(dirname):
    l = dirname.split("/")
    for i in range(len(l)):
        create_folder_if_necessary("/".join(l[:i+1]))

def create_file_path_if_necessary(filepath):
    if "/" in filepath:
        create_path_if_necessary(filepath[:filepath.rfind("/")])

def get_relative_error(position, target, size):
    """Returns the relative error between the current robot position
    and its target
    0 means that the target is reached
    1 means that the robot is at the furthest possible position from the target"""
    ex = (position[0] - target[0]) / size
    ey = (position[1] - target[1]) / size
    etheta = (position[2] - target[2]) / math.pi  #Faut trouver la bonne relation pour etheta
    if etheta<=-1:
       etheta+=2
    elif etheta>1:
       etheta-=2
    relative_error = math.sqrt(ex**2 / 3 + ey**2 / 3 + etheta**2 / 3)
    return relative_error


def change_repere_x_robot_vers_NN(x,y,theta):
    return x*cos(theta)+y*sin(theta)

def change_repere_y_robot_vers_NN(x,y,theta):
    return y*cos(theta)-x*sin(theta)


def change_repere_robot_vers_NN(position,theta):
    new_position=[0,0,0]
    new_position[0]=change_repere_x_robot_vers_NN(position[0],position[1],theta)
    new_position[1]=change_repere_y_robot_vers_NN(position[0],position[1],theta)
    new_position[2]=position[2]-theta
    return new_position

def change_repere_commande(command,theta):
    new_command=[0,0]
    new_command[0]=command[0]*cos(theta)-sin(theta)*command[1]
    new_command[1]=command[1]*cos(theta)+sin(theta)*command[0]
    return new_command

def sigmoid(x,param):
    return (2*math.exp(param*x)/(math.exp(param*x)+1))-1

def smooth(com,last_com,smoothness):
    smooth_com=com
    if com[0]-last_com[0]>smoothness:
        smooth_com[0]=last_com[0]+smoothness
    if com[1]-last_com[1]>smoothness:
        smooth_com[1]=last_com[1]+smoothness
    if com[0]-last_com[0]<-1*smoothness:
        smooth_com[0]=last_com[0]-smoothness
    if com[1]-last_com[1]<-1*smoothness:
        smooth_com[1]=last_com[1]-smoothness
    return smooth_com

def modulo(theta):
    return (theta+pi)%(2*pi)-pi

#Il faut ajouter devant les atan un paramètres pour atténuer les oscillations
def theta_s(x,y, ratio, experimental_theta,com,theta):
    """Returns the theta shift of the robot
    The goal of this shift if to prevent the robot from being stuck
    when it reaches a local minimum of the standard error"""
    #return ((-pi/2)+math.atan(ratio*x)/(1+abs(y)))*(((2*exp(2*y))/(1+exp(2*y))-1))*(0+1.0*math.sqrt(com[0]**2+com[1]**2))+orientation_max(y)*(1-0.5*math.sqrt(com[0]**2+com[1]**2))*((2*exp(1.5*abs(y))/(1+exp(1.5*abs(y)))-1))*(-pi/2)
    #return sigmoid(y,3)*sigmoid(abs(x),2)*(-pi/2)

    
    #x,y=change_repere_x_robot_vers_NN(x,y,theta),change_repere_y_robot_vers_NN(x,y,theta)
    t=0


    #if abs(y)<0.03:
    #    t=sigmoid(y/x,0.1)*(pi/2)
    if x>0:
        t= math.atan(y/x)
    if x<0:
        t= (pi+math.atan(y/x))%(2*pi)-pi
    if x==0:
        t=0
    if sqrt(x**2+y**2)<0.1:
        t=t*sigmoid(y,2)*sigmoid(x,2)
    t=t-theta
    if theta>pi/2 or theta<-pi/2:
        t=(t)%(2*pi)-pi
    else:
        t=(t+pi)%(2*pi)-pi
    if x !=0 and y !=0:
        t=t*exp(-1/abs(500*x*y))
    
    return t
"""
    if not experimental_theta:
        if x>0:
            return math.atan(ratio*y)
        if x<=0:
            return math.atan(-ratio*y)
    else:
        if x>0:
            t = math.atan(ratio*y)
        else:
            t = -math.atan(ratio*y)
"""

        #size = 4
        #m = 0.4
        #rate = 0.50
        #X = -m/log((1 - rate))
        #d = abs(x)
        #c = exp(-d / X)
        #c = 1 + c
        #e = 1 - exp(-(sqrt(x**2+y**2) / 0.40))
        #return t        
        #return c*t * e

def deep_copy(l):
    if isinstance(l, list):
        return list(map(deep_copy, l))
    return l

class OnlineTrainer:
    def __init__(self, robot, NN):
        """
        Args:
            robot (Robot): a robot instance following the pattern of
                VrepPioneerSimulation
            target (list): the target position [x,y,theta]
        """
        self.robot = robot
        self.network = NN

        self.alpha = [1/4,1/4,1/((math.pi))]

        self.ready_to_exit = False

    def train(self, target, options={}, tick=0.050, prediction=False, experimental_theta=False, gain=1, restrict_theta_shift=True, restrict_propagation=False, invert_restriction=False, random_ratio=0, learning_step=0.05, theta_shift_ratio=10, size=4, maximum_duration=0, stop_criterion=None, verbose=False, log='', neuron_file_name='', description="No Description"):

        self.ready_to_exit = False

        self.alpha = [1/size,1/size,1/((math.pi))]
        #target=change_repere_robot_vers_NN(target,target[2])

        # This coefficient can reduce the theta input to [-0.75, +0.75]
        self.alpha[2] *= 1

        position = self.robot.get_position() # This function takes 50 ms to compute
        #position=change_repere_robot_vers_NN(position,target[2])


        self.stop_automatically = stop_criterion!=None
        if self.stop_automatically:
            self.percent_criterion = stop_criterion[0]
            self.duration_criterion = stop_criterion[1]
        self.stop_criterion_reached = False

        network_input = [0, 0, 0]
        network_input[0] = (position[0]-target[0])*self.alpha[0]
        network_input[1] = (position[1]-target[1])*self.alpha[1]
        network_input[2] = modulo((position[2]-target[2]))*self.alpha[2]
        

        t0 = time.time()

        if log or neuron_file_name:
            times = []
        if log:
            positions = []
            commands = []
            criterions = []
            gradients = []
            theta_shifts = []
        if neuron_file_name:
            neurons_input = []
            neurons_output = []

        criterion_reach_date = time.time()


# delta_t = tick n'est pas réaliste, il faudrait calculer le delta_t
        delta_t = tick

        relative_error = 1000

        t_ref = time.time()

        xth=0
        yth=0
        thetath=0
        command=[0,0]
        last_command=[0,0]

        # Loop until one of the ending criterions is met
        
        while self.running and not (self.stop_automatically and self.stop_criterion_reached):
            new_system = True

            t = time.time()

           


            # Get the robot new position
            position = self.robot.get_position() # this takes 50 ms to process
            #position=change_repere_robot_vers_NN(position,target[2])

            new_error = get_relative_error(position, target, size)

            error_is_smaller = new_error < relative_error
            #12/12 essai en passant de .20 à .40
            enable_theta_shift = (error_is_smaller or new_error > 0.40)

            # Compute the new input values for the Neuron Network
            if (not restrict_theta_shift) or enable_theta_shift:
                network_input[2]=modulo((((position[2]-target[2]-theta_s(position[0]-target[0], position[1]-target[1], theta_shift_ratio, experimental_theta,command,target[2])+math.pi)%(2*math.pi))-math.pi))*self.alpha[2]
            else:
                network_input[2]=porte_theta((position[2]-target[2]+math.pi)%(2*math.pi)-math.pi)*self.alpha[2]
            info = ", t_s=" + str(int(theta_s(position[0]-target[0], position[1]-target[1], theta_shift_ratio, experimental_theta,command,target[2])*360/2/math.pi))
            network_input[1] = (position[1]-target[1])*self.alpha[1]
            network_input[0] = (position[0]-target[0])*self.alpha[0]

            if log:
                theta_shifts.append(theta_s(position[0]-target[0], position[1]-target[1], theta_shift_ratio, experimental_theta,command,target[2]))

            # Tell the Neuron Network to make an iteration (propagation)
            #print("input " + str(network_input[2]))
            
            command = [i*1 for i in self.network.runNN(network_input)]
            #change_repere_commande(command,target[2])
            command=smooth(command,last_command,0.15)
            last_command=command

# ici il faudrait que le calcul du gradient intervienne
# de manière à avoir le gradient avant d'effectuer le mouvement
# pour pouvoir mieux corriger nos poids lors de l'apprentissage du NN



            if log or neuron_file_name:
                times.append(t - t0)
            if log:
                # Save the current state of the lesson
                positions.append(self.robot.get_position()) # This function takes 50 ms to compute
                commands.append(command[:])
                criterions.append(get_relative_error(position, target, size))
            if neuron_file_name:
                neurons_input.append(deep_copy(self.network.wi))
                neurons_output.append(deep_copy(self.network.wo))

            if self.training:

                if prediction:

                    r = self.robot.r
                    R = self.robot.R
                    x = position[0]
                    y = position[1]
                    theta = position[2]
                    x_target = target[0]
                    y_target = target[1]
                    if (not restrict_theta_shift) or enable_theta_shift or self.training:
                        theta_target = ((target[2] + theta_s(position[0] - target[0], position[1] - target[1], theta_shift_ratio, experimental_theta,command,target[2]) + math.pi)%(2 * math.pi)) - math.pi # With theta_shift
                    else:
                        theta_target = ((target[2] + math.pi)%(2 * math.pi)) - math.pi # Without theta_shift


# calcul de Xt+1 et Yt+1

                    new_delta_t = time.time() - t_ref
                    t_ref = time.time()




                    xth=2*pi*(r/2)*(command[0]+command[1])*cos(theta)*new_delta_t+x
                    yth=2*pi*(r/2)*(command[0]+command[1])*sin(theta)*new_delta_t+y
                    str1="x="+str(x)
                    #print(str1)
                    str2="xth="+str(xth)
                    #print(str2)
                    thetath=(4*r/(R))*(command[1]-command[0])*new_delta_t+theta


                    grad = get_grad(self, r, R, size, xth, yth, theta, thetath, x_target, y_target, theta_target, new_delta_t)
                    #print("grad")
                    #print(grad)
                    grad = [new_delta_t * gain * grad[0], new_delta_t * gain * grad[1]]
                    #print(grad)

                else: # Old method

                    grad = [
                        ((-1)/(delta_t**2))*(network_input[0]*delta_t*self.robot.r*math.cos(position[2])
                        +network_input[1]*delta_t*self.robot.r*math.sin(position[2])
                        -network_input[2]*delta_t*self.robot.r/(2*self.robot.R)),

                        ((-1)/(delta_t**2))*(network_input[0]*delta_t*self.robot.r*math.cos(position[2])
                        +network_input[1]*delta_t*self.robot.r*math.sin(position[2])
                        +network_input[2]*delta_t*self.robot.r/(2*self.robot.R))
                    ]

                if log:
                    gradients.append(grad)

                if (error_is_smaller and not invert_restriction) or (invert_restriction and not error_is_smaller):
                    # The two args after grad are the gradient learning steps for t
                    # and t-1
                    self.network.backPropagate(grad, learning_step, 0)
                elif random_ratio > 0:
                    self.network.random_update(random_ratio)
                elif not restrict_propagation:
                    self.network.backPropagate(grad, learning_step, 0)

            # Change the robot wheel velocity accordingly
            #print(command)
            self.robot.set_motor_velocity(command) # this takes 100 ms to process

            # Compute the relative error
            relative_error = get_relative_error(position, target, size)

            # Display the current data
            s = "[" + str(round(t-t0, 1)) + " sec.] x=" + str(round(position[0], 3)) + ", y=" + str(round(position[1], 3)) + ", th=" + str(int(position[2]*360/2/math.pi)) + ", err=" + str(round(100*relative_error, 1)) + "%" + info
            if len(s) < 60:
                s += " "*(60-len(s))
            if self.running:
                if verbose:
                    print(s)
                else:
                    print(s, end="\r")

            if maximum_duration > 0 and t >= t0 + maximum_duration:
                # Maximum duration reached : the lesson is forced to stop
                self.running = False
            elif self.stop_automatically:
                # Check if the robot has converged
                if relative_error <= self.percent_criterion:
                    if t >= criterion_reach_date + self.duration_criterion:
                        self.stop_criterion_reached = True
                else:
                    criterion_reach_date = t

            # Wait to prevent underflow
            t_after_iteration = time.time()
            if tick > t_after_iteration - t:
                time.sleep(tick - (t_after_iteration - t))

        # End of the lesson
        self.robot.set_motor_velocity([0,0]) # This function takes 100 ms to compute
        self.running = False

        if log:
            # Save the logs in a file
            duration = time.time() - t0
            obj = {"options":dict(options), "target":target, "duration":duration, "times":times, "positions":positions, "commands":commands, "criterions":criterions, "gradients":gradients, "theta_shifts":theta_shifts}
            create_file_path_if_necessary("logs/"+log)
            with open("logs/"+log, 'w') as f:
                json.dump(obj, f)
                f.close()
            autosave_name = "logs/AUTOSAVE/"+str(int(time.time()))+".json"
            create_file_path_if_necessary(autosave_name)
            with open(autosave_name, 'w') as f:
                json.dump(obj, f)
                f.close()

        if neuron_file_name:
            obj = {"times":times, "input":neurons_input, "output":neurons_output}
            create_file_path_if_necessary("networks/"+log)
            with open("networks/"+log, 'w') as f:
                json.dump(obj, f)
                f.close()

        self.ready_to_exit = True

