import numpy as np 
import os 
import math 
import random as rand 
import logz
import time

#Verified
def random_controller(min_action=40,max_action=90): #the range is starting from 0 volts to 90 volts
    return rand.uniform(min_action,max_action)


#Verified
# Utils Functions
def sgn(x):
    return x/abs(x)

def Abs_Error(x,xd):
    return abs(x-xd)

#Verified
# The inputs X and u should be numpy arrays
def leader_cost_function(X,u,X_desired=[0,10],vehicle_type='Leader'):
    error=[]
    for c in range(len(X)):
        if(vehicle_type=='Leader'):
            error.append(Abs_Error(X[c][1],X_desired[1]))
        elif (vehicle_type=='Follower'):
            error.append(Abs_Error(X[c][0],X_desired[0]))
    return np.sum(np.array(error))

#Verified
def cost_evaluate(cost_function,imag_rollouts,X_desired=[0,10],vehicle_type='Leader'):
    costs=[]
    for i in range(imag_rollouts.shape[0]):
        costs.append(cost_function(imag_rollouts[i,0,:],imag_rollouts[i,1,:],X_desired,vehicle_type))
    return np.array(costs)


#Verified
def mpc_like_controller(vehicle_model, pred_horizon, imag_rollouts_number,X_current,cost_function,X_desired=[0,10],vehicle_type='Leader',X_leader=None):
    imag_rollouts=[]
    for i in range(imag_rollouts_number):
        actions=[]
        states=[]
        states_follower=[]
        X_prev=X_current
        for c in range(pred_horizon):
            action=random_controller()
            if(vehicle_type=="Follower"):
                states_follower.append([X_leader[0]-X_prev[0],X_leader[1]-X_prev[1]])
            else:
                states.append(X_prev)
            actions.append(action)
            X_prev=vehicle_model(action,X_prev[0],X_prev[1])

        if(vehicle_type=="Follower"):
            imag_rollouts.append([states_follower,actions])
        else:
            imag_rollouts.append([states,actions])

    imag_rollouts=np.asarray(imag_rollouts)
    # if(vehicle_type=='Follower'):
    #     print(cost_evaluate(cost_function,imag_rollouts,X_desired,vehicle_type))

    least_cost_index=np.argmin(cost_evaluate(cost_function,imag_rollouts,X_desired,vehicle_type))
    
    return imag_rollouts[least_cost_index,1,0]



def mpc_sampler(vehicle_model,controller,cost_function,save,timeSteps=2000,pred_horizon=15, imag_rollouts_number=100,X_initial=[0,0],X_desired=[0,10]):
    X=[]
    u=[]
    cost=[]
    X.append(X_initial)
    cost.append(Abs_Error(X_initial[1],X_desired[1]))
    if(save):
        logz.configure_output_dir("/home/hendawy/Desktop/Platoon_Advanced_Mechatronics_Project/RLTrial",0)
    for t in range(timeSteps):
        # time_start=time.time()
        u_t=controller(vehicle_model,pred_horizon,imag_rollouts_number,X[t],cost_function)
        X_next=vehicle_model(u_t,X[t][0],X[t][1])
        cost.append(Abs_Error(X_next[1],X_desired[1]))
        if(save):
            logz.log_tabular('Error', Abs_Error(X_next[1],X_desired[1]))
            logz.dump_tabular()
        X.append(X_next)
        u.append([u_t])
        # time_end=time.time()
        # print(time_end-time_start)
    X.pop()
    traj = {"states" : np.array(X), 
            "control" : np.array(u),
            "cost" : np.array(cost),
            }
    return traj


#Verified
def vehicle_model(u_drive,
                  X1_prev,
                  X2_prev,
                  delta_t=0.0186/2,
                  Kt=0.5791,
                  Kv=0.573,
                  R=1.45,
                  S=0.005,
                  Lamda=6.78*1e-4,
                  r=0.2775,
                  Cd=0.4,
                  rho=1.18,
                  A=1.353,
                  Vw=1,
                  m=100,
                  J=0.002,
                  g=9.81,
                  D=1,
                  C=1.9,
                  B=10,
                  E=0.97,
                  ):
    
    H_magic=D*math.sin(C*(math.atan(B*S-E*(B*S-(math.atan(B*S))))))
    a_1=(2*Kt*Kv*(S+1))/(R*math.pow(r,2))
    a_2=(2*Lamda*(S+1))/(math.pow(r,2))
    a_3=0.5*Cd*rho*A*sgn(X2_prev+Vw)
    X2_dot=(1/(m+(2*J*(S+1))/math.pow(r,2)))*((2*Kt/(R*r))*u_drive-(a_1+a_2+2*Vw*a_3)*X2_prev-a_3*math.pow(X2_prev,2)-math.pow(Vw,2)*a_3-m*g*H_magic)
    X2_next=X2_dot*delta_t+X2_prev
    X1_dot=X2_next
    X1_next=X1_dot*delta_t+X1_prev
    X=[X1_next,X2_next]
    return X


def platoon_model(u_drive_1,u_drive_2,X1_prev_1,X2_prev_1,X1_prev_2,X2_prev_2):
    X_1=vehicle_model(u_drive_1,X1_prev_1,X2_prev_1)
    X_2=vehicle_model(u_drive_2,X1_prev_2,X2_prev_2)
    print(X_2)
    X_2[0]=X_1[0]-X_2[0]
    X_2[1]=X_1[1]-X_2[1]
    return X_1,X_2

def mpc_platoon_sampler(vehicle_model,platoon_model,controller,cost_function,save,timeSteps=3000,pred_horizon=15, imag_rollouts_number=400,X_initial_1=[0,0],X_initial_2=[0,0],X_desired_1=[0,10],X_desired_2=[2,0]):
    X_1=[]
    X_2=[]
    X_v2=[]
    u_1=[]
    u_2=[]
    cost_1=[]
    cost_2=[]
    X_1.append(X_initial_1)
    X_v2.append(X_initial_2)
    X_2.append([X_1[0][0]-X_v2[0][0],X_1[0][1]-X_v2[0][1]])
    cost_1.append(Abs_Error(X_initial_1[1],X_desired_1[1]))
    cost_2.append(Abs_Error(X_2[0][0],X_desired_2[0]))
    if(save):
        logz.configure_output_dir("/home/hendawy/Desktop/Platoon_Advanced_Mechatronics_Project/RLTrial",11)
    for t in range(timeSteps):
        # time_start=time.time()
        u1_t=controller(vehicle_model,pred_horizon,imag_rollouts_number,X_1[t],cost_function,X_desired_1,'Leader')
        u2_t=controller(vehicle_model,pred_horizon,imag_rollouts_number,X_v2[t],cost_function,X_desired_2,'Follower',X_1[t])
        X_next_1,X_next_2=platoon_model(u1_t,u2_t,X_1[t][0],X_1[t][1],X_v2[t][0],X_v2[t][1])
        # print('Vehicle 1',X_1[t],X_next_1,u1_t)
        # print('Vehicle 2',X_v2[t],X_next_2,u2_t)
        cost_1.append(Abs_Error(X_next_1[1],X_desired_1[1]))
        cost_2.append(Abs_Error(X_next_2[0],X_desired_2[0]))
        if(save):
            logz.log_tabular('Error_v1', Abs_Error(X_next_1[1],X_desired_1[1]))
            logz.log_tabular('Error_v2', Abs_Error(X_next_2[0],X_desired_2[0]))
            logz.dump_tabular()
        X_1.append(X_next_1)
        X_2.append(X_next_2)
        u_1.append([u1_t])
        u_2.append([u2_t])
        X_v2.append([-X_next_2[0]+X_next_1[0],-X_next_2[1]+X_next_1[1]])
        # time_end=time.time()
        # print(time_end-time_start)
    X_1.pop()
    X_2.pop()
    traj = {"states_v1" : np.array(X_1),
            "states_v2" : np.array(X_v2),
            "states_f1" : np.array(X_2),
            "control_v1" : np.array(u_1),
            "control_v2" : np.array(u_2),
            "cost_v1" : np.array(cost_1),
            "cost_v2" : np.array(cost_2),
            }
    return traj

def main():
    mpc_platoon_sampler(vehicle_model,platoon_model,mpc_like_controller,leader_cost_function,save=True,
                        timeSteps=2000,pred_horizon=15, imag_rollouts_number=100,
                        X_initial_1=[5,0],X_initial_2=[0,0],X_desired_1=[0,10],X_desired_2=[3,0])

if __name__ == "__main__":
    main()



