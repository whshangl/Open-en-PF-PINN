"""Backend supported: pytorch"""
import deepxde as dde
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

dde.config.set_random_seed(2024)
dde.config.set_default_float('float64')

device = torch.device("cuda:0")
torch.set_default_device(device)

torch.cuda.empty_cache()

### Problem parameters:
domain_length = 4

L_norm = domain_length/2

a1 = -0.1725
kappa = 0.58
a11 = -0.073
a12 = 0.75
a111 = 0.26
a112 = 0.61
a123 = -3.7
c11 = 174.6
c12 = 79.37
c44 = 111.1
q11 = 11.41
q12 = 0.4607
q44 = 7.499
G11 = 0.10
G12 = 0
G44 = 0.05
G44_= 0.05

ab11 = 0.42286 
ab12 = 0.73541



### governing equations
def pde(X, Y):
    """
    Expresses the PDE of the phase-field model. 
    Argument X to pde(X,Y) is the input, where X[:, 0] is x-coordinate, X[:,1] is y-coordinate.
    Argument Y to pde(X,Y) is the output, with 5 variables u1, u2, phi, P1, P2, as shown below.
    """
    u1  = Y[:, 0:1]   ## displacement in 1-direction
    u2  = Y[:, 1:2]   ## displacement in 2-direction
    phi = Y[:, 2:3]   ## electric potential 
    P1  = Y[:, 3:4]   ## polarization in 1-direction
    P2  = Y[:, 4:5]    ## polarization in 2-direction

    u1_x  = dde.grad.jacobian(Y, X, i = 0, j = 0)   ## \frac{\partial{u1}}{\partial{x}}
    u2_x  = dde.grad.jacobian(Y, X, i = 1, j = 0)   ## \frac{\partial{u2}}{\partial{x}}
    phi_x = dde.grad.jacobian(Y, X, i = 2, j = 0)   ## \frac{\partial{phi}}{\partial{x}}
    P1_x  = dde.grad.jacobian(Y, X, i = 3, j = 0)   ## \frac{\partial{P1}}{\partial{x}}
    P2_x  = dde.grad.jacobian(Y, X, i = 4, j = 0)   ## \frac{\partial{P2}}{\partial{x}}

    u1_y  = dde.grad.jacobian(Y, X, i = 0, j = 1)   ## \frac{\partial{u1}}{\partial{y}}
    u2_y  = dde.grad.jacobian(Y, X, i = 1, j = 1)   ## \frac{\partial{u2}}{\partial{y}}
    phi_y = dde.grad.jacobian(Y, X, i = 2, j = 1)   ## \frac{\partial{phi}}{\partial{y}}
    P1_y  = dde.grad.jacobian(Y, X, i = 3, j = 1)   ## \frac{\partial{P1}}{\partial{y}}
    P2_y  = dde.grad.jacobian(Y, X, i = 4, j = 1)   ## \frac{\partial{P2}}{\partial{y}}

    u1_xx   = dde.grad.hessian(Y, X, component= 0, i = 0, j = 0)  ## \frac{\partial^2{u1}}{\partial{x}^2}
    u2_xx   = dde.grad.hessian(Y, X, component= 1, i = 0, j = 0)  ## \frac{\partial^2{u2}}{\partial{x}^2}
    phi_xx  = dde.grad.hessian(Y, X, component= 2, i = 0, j = 0)  ## \frac{\partial^2{phi}}{\partial{x}^2}
    P1_xx   = dde.grad.hessian(Y, X, component= 3, i = 0, j = 0)  ## \frac{\partial^2{P1}}{\partial{x}^2}
    P2_xx   = dde.grad.hessian(Y, X, component= 4, i = 0, j = 0)  ## \frac{\partial^2{P2}}{\partial{x}^2}

    u1_yy   = dde.grad.hessian(Y, X, component= 0, i = 1, j = 1)  ## \frac{\partial^2{u1}}{\partial{y}^2}
    u2_yy   = dde.grad.hessian(Y, X, component= 1, i = 1, j = 1)  ## \frac{\partial^2{u2}}{\partial{y}^2}
    phi_yy  = dde.grad.hessian(Y, X, component= 2, i = 1, j = 1)  ## \frac{\partial^2{phi}}{\partial{y}^2}
    P1_yy   = dde.grad.hessian(Y, X, component= 3, i = 1, j = 1)  ## \frac{\partial^2{P1}}{\partial{y}^2}
    P2_yy   = dde.grad.hessian(Y, X, component= 4, i = 1, j = 1)  ## \frac{\partial^2{P2}}{\partial{y}^2}

    u1_xy   = dde.grad.hessian(Y, X, component= 0, i = 0, j = 1)  ## \frac{\partial^2{u1}}{\partial{x}\partial{y}}
    u2_xy   = dde.grad.hessian(Y, X, component= 1, i = 0, j = 1)  ## \frac{\partial^2{u2}}{\partial{x}\partial{y}}
    phi_xy  = dde.grad.hessian(Y, X, component= 2, i = 0, j = 1)  ## \frac{\partial^2{phi}}{\partial{x}\partial{y}}
    P1_xy   = dde.grad.hessian(Y, X, component= 3, i = 0, j = 1)  ## \frac{\partial^2{P1}}{\partial{x}\partial{y}}
    P2_xy   = dde.grad.hessian(Y, X, component= 4, i = 0, j = 1)  ## \frac{\partial^2{P2}}{\partial{x}\partial{y}}


    
    ###############################################################
    ### div(sigma) = 0 related expressions
    ### strain: plane strain assumption is used, i.e., epsilon33 = epsilon13 = epsilon23 = 0
    epsilon11_ = u1_x/L_norm   
    epsilon22_ = u2_y/L_norm
    epsilon12_ = 0.5 * (u1_y + u2_x)/L_norm

    epsilon11_x_ = u1_xx/L_norm/L_norm
    epsilon11_y_ = u1_xy/L_norm/L_norm

    epsilon12_y_ = 0.5 * (u1_yy + u2_xy)/L_norm/L_norm
    epsilon12_x_ = 0.5 * (u1_xy + u2_xx)/L_norm/L_norm

    epsilon22_x_ = u2_xy/L_norm/L_norm
    epsilon22_y_ = u2_yy/L_norm/L_norm
    
    P1_x_ = P1_x/L_norm
    P2_x_ = P2_x/L_norm
    P1_y_ = P1_y/L_norm
    P2_y_ = P2_y/L_norm
    
    P1_xx_ = P1_xx/L_norm/L_norm
    P1_yy_ = P1_yy/L_norm/L_norm
    P1_xy_ = P1_xy/L_norm/L_norm
    
    P2_xx_ = P2_xx/L_norm/L_norm
    P2_yy_ = P2_yy/L_norm/L_norm
    P2_xy_ = P2_xy/L_norm/L_norm
    

    ### stress
    sigma11 = c11 * epsilon11_ + c12 * epsilon22_ - q11 * P1 * P1 - q12 * P2 * P2
    sigma22 = c11 * epsilon22_ + c12 * epsilon11_ - q11 * P2 * P2 - q12 * P1 * P1
    sigma12 = 2 * c44 * epsilon12_ - q44 * P1 * P2
    
    ### divergence of stress
    sigma11_x = c11 * epsilon11_x_ + c12 * epsilon22_x_ - 2 * q11 * P1 * P1_x_ - 2 * q12 * P2 * P2_x_
    sigma12_y = 2 * c44 * epsilon12_y_ - q44 * P2 * P1_y_ - q44 * P1 * P2_y_
    sigma12_x = 2 * c44 * epsilon12_x_ - q44 * P2 * P1_x_ - q44 * P1 * P2_x_
    sigma22_y = c11 * epsilon22_y_ +  c12 * epsilon11_y_ - 2 * q11 * P2 * P2_y_ - 2 * q12 * P1 * P1_y_
    
    ###############################################################
    ### div(D) = 0 related expressions
    ### electric field
    E1_ = -phi_x/L_norm
    E2_ = -phi_y/L_norm

    E1_x_ = -phi_xx/L_norm/L_norm
    E2_y_ = -phi_yy/L_norm/L_norm

    ### electric displacement
    D1 = kappa * E1_ + P1
    D2 = kappa * E2_ + P2

    ### divergence of electric displacement
    D1_x = kappa * E1_x_ + P1_x_
    D2_y = kappa * E2_y_ + P2_y_

    ###############################################################
    ### TDGL equation related expressions
    ### h_P1 = \frac{\partial{h}}{\partial{P1}}
    h_P1 = + 2 * a1 * P1 \
           + 4 * ab11 * (P1**3)  \
           + 6 * a111 * (P1**5) \
           - 2 * q11 * epsilon11_ * P1 - 2 * q12 * P1 * epsilon22_ - 2 * q44 * epsilon12_ * P2 \
           - E1_ \
           + 2 * ab12 * P1 * (P2**2) \
           + 4 * a112 * (P1**3) * (P2**2) + 2 * a112 * P1 * (P2**4) \

    
    ### h_P2 = \frac{\partial{h}}{\partial{P2}}
    h_P2 = + 2 * a1 * P2 \
           + 4 * ab11 * (P2**3)  \
           + 6 * a111 * (P2**5) \
           - 2 * q11 * epsilon22_ * P2 - 2 * q12 * P2 * epsilon11_ - 2 * q44 * epsilon12_ * P1 \
           - E2_ \
           + 2 * ab12 * P2 * (P1**2) \
           + 4 * a112 * (P2**3) * (P1**2) + 2 * a112 * P2 * (P1**4) \
    
    ### chi_{ij} = \frac{\partial{h}}{\partial{xi_{ij}}}, xi_{ij} = \frac{\partial{P_i}}{\partial{x_j}}
    chi11 = G11 * P1_x_ + G12 * P2_y_
    chi12 = G44 * (P1_y_ + P2_x_) + G44_ * (P1_y_ - P2_x_)
    chi21 = G44 * (P1_y_ + P2_x_) + G44_ * (P2_x_ - P1_y_)
    chi22 = G11 * P2_y_ + G12 * P1_x_

    ### divergence of chi_{ij}
    chi11_x = G11 * P1_xx_ + G12 * P2_xy_
    chi12_y = G44 * (P1_yy_ + P2_xy_) + G44_ * (P1_yy_ - P2_xy_)
    chi21_x = G44 * (P1_xy_ + P2_xx_) + G44_ * (P2_xx_ - P1_xy_)
    chi22_y = G11 * P2_yy_ + G12 * P1_xy_

    ### divergence of {chi_{ij}}_{2*2}
    div_P1 = chi11_x + chi12_y 
    div_P2 = chi21_x + chi22_y

    ###############################################################
    ### balance equations
    balance_mechanic_1 = sigma11_x + sigma12_y
    balance_mechanic_2 = sigma12_x + sigma22_y

    balance_electric = D1_x + D2_y

    TDGL_1 = h_P1 - div_P1
    TDGL_2 = h_P2 - div_P2
    

    ### different energy
    h_elastic =  0.5 * c11 * (epsilon11_ * epsilon11_ + epsilon22_ * epsilon22_) \
               + c12 * (epsilon11_ * epsilon22_ ) \
               + 2 * c44 * (epsilon12_ * epsilon12_ )
    
    h_coupling = -q11 * (epsilon11_ * P1 * P1 + epsilon22_ * P2 * P2) \
                 -q12 * (epsilon11_ * P2 * P2 + epsilon22_ * P1 * P1) \
                 -2 * q44 * (epsilon12_ * P1 * P2)
    
    h_electrostatic = -0.5 * kappa * (E1_ * E1_ + E2_ * E2_) - E1_ * P1 - E2_ * P2

    f_electric = 0.5 * kappa * (E1_ * E1_ + E2_ * E2_) 

    h_Landau = + a1 * (P1 * P1 + P2 * P2) + ab11 * (P1**4 + P2**4) + ab12 * (P1 * P1 * P2 * P2) \
               + a111 * (P1**6 + P2**6) + a112 * ((P1**4)*(P2 * P2) + (P2**4)*(P1 * P1))
    
    h_gradient = + 0.5 * G11 * (P1_x_ * P1_x_ + P2_y_ * P2_y_) + G12 * (P1_x_ * P2_y_) \
                 + 0.5 * G44 * (P1_y_ + P2_x_) * (P1_y_ + P2_x_)  + 0.5 * G44_ * (P1_y_ - P2_x_) * (P1_y_ - P2_x_) 
    

    
    sum_energy = (h_elastic + h_coupling + f_electric + h_Landau + h_gradient) * (domain_length * domain_length)*1
    
    return [balance_mechanic_1, balance_mechanic_2, balance_electric, TDGL_1, TDGL_2, sum_energy]

### Computational geometry:
geom = dde.geometry.Rectangle(xmin=[-1*domain_length/2/L_norm, -1*domain_length/2/L_norm], xmax=[domain_length/2/L_norm, domain_length/2/L_norm])


def boundary_left_right(X, on_boundary):
    return on_boundary and (np.isclose(X[0], -1*domain_length/2/L_norm) or np.isclose(X[0], domain_length/2/L_norm))

def boundary_bottom_top(X, on_boundary):
    return on_boundary and (np.isclose(X[1], -1*domain_length/2/L_norm) or np.isclose(X[1], domain_length/2/L_norm))

def boundary_all(X, on_boundary):
    return on_boundary and (np.isclose(X[0], -1*domain_length/2/L_norm) or np.isclose(X[0], domain_length/2/L_norm) or np.isclose(X[1], -1*domain_length/2/L_norm) or np.isclose(X[1], domain_length/2/L_norm))


def boundary_flux(X,Y):
    u1  = Y[:, 0:1]   ## displacement in 1-direction
    u2  = Y[:, 1:2]   ## displacement in 2-direction
    phi = Y[:, 2:3]   ## electric potential 
    P1  = Y[:, 3:4]   ## polarization in 1-direction
    P2  = Y[:, 4:5]   ## polarization in 2-direction

    u1_x  = dde.grad.jacobian(Y, X, i = 0, j = 0)   ## \frac{\partial{u1}}{\partial{x}}
    u2_x  = dde.grad.jacobian(Y, X, i = 1, j = 0)   ## \frac{\partial{u2}}{\partial{x}}
    phi_x = dde.grad.jacobian(Y, X, i = 2, j = 0)   ## \frac{\partial{phi}}{\partial{x}}
    P1_x  = dde.grad.jacobian(Y, X, i = 3, j = 0)   ## \frac{\partial{P1}}{\partial{x}}
    P2_x  = dde.grad.jacobian(Y, X, i = 4, j = 0)   ## \frac{\partial{P2}}{\partial{x}}

    u1_y  = dde.grad.jacobian(Y, X, i = 0, j = 1)   ## \frac{\partial{u1}}{\partial{y}}
    u2_y  = dde.grad.jacobian(Y, X, i = 1, j = 1)   ## \frac{\partial{u2}}{\partial{y}}
    phi_y = dde.grad.jacobian(Y, X, i = 2, j = 1)   ## \frac{\partial{phi}}{\partial{y}}
    P1_y  = dde.grad.jacobian(Y, X, i = 3, j = 1)   ## \frac{\partial{P1}}{\partial{y}}
    P2_y  = dde.grad.jacobian(Y, X, i = 4, j = 1)   ## \frac{\partial{P2}}{\partial{y}}


    ### strain: plane strain assumption is used, i.e., epsilon33 = epsilon13 = epsilon23 = 0
    epsilon11_ = u1_x/L_norm  
    epsilon22_ = u2_y/L_norm
    epsilon12_ = 0.5 * (u1_y + u2_x)/L_norm
  
    ### stress
    sigma11 = c11 * epsilon11_ + c12 * epsilon22_ - q11 * P1 * P1 - q12 * P2 * P2
    sigma22 = c11 * epsilon22_ + c12 * epsilon11_ - q11 * P2 * P2 - q12 * P1 * P1
    sigma12 = 2 * c44 * epsilon12_ - q44 * P1 * P2

    ### electric field
    E1_ = -phi_x/L_norm
    E2_ = -phi_y/L_norm

    ### electric displacement
    D1 = kappa * E1_ + P1
    D2 = kappa * E2_ + P2

    ### chi_{ij} = \frac{\partial{h}}{\partial{xi_{ij}}}, xi_{ij} = \frac{\partial{P_i}}{\partial{x_j}}
    
    P1_x_ = P1_x/L_norm
    P1_y_ = P1_y/L_norm
    
    P2_x_ = P2_x/L_norm
    P2_y_ = P2_y/L_norm
    
    chi11 = G11 * P1_x_ + G12 * P2_y_
    chi12 = G44 * (P1_y_ + P2_x_) + G44_ * (P1_y_ - P2_x_)
    chi21 = G44 * (P1_y_ + P2_x_) + G44_ * (P2_x_ - P1_y_)
    chi22 = G11 * P2_y_ + G12 * P1_x_

    return [sigma11, sigma22, sigma12, D1, D2, chi11, chi22, chi12, chi21]


#### boundary condition: sigma*n = 0 (traction free)
bc_LeftRight_traction_11 = dde.icbc.OperatorBC(geom, lambda X, Y, _: boundary_flux(X,Y)[0], boundary_left_right)
bc_BottomTop_traction_22 = dde.icbc.OperatorBC(geom, lambda X, Y, _: boundary_flux(X,Y)[1], boundary_bottom_top)
bc_All_traction_12       = dde.icbc.OperatorBC(geom, lambda X, Y, _: boundary_flux(X,Y)[2], boundary_all)

### boundary condition: D*n = 0 (surface charge free)
bc_LeftRight_charge_1 = dde.icbc.OperatorBC(geom, lambda X, Y, _: boundary_flux(X,Y)[3], boundary_left_right)
bc_BottomTop_charge_2 = dde.icbc.OperatorBC(geom, lambda X, Y, _: boundary_flux(X,Y)[4], boundary_bottom_top)

### boundary condition: surface gradient flux free
bc_LeftRight_gradient_11 = dde.icbc.OperatorBC(geom, lambda X, Y, _: boundary_flux(X,Y)[5], boundary_left_right)
bc_LeftRight_gradient_21 = dde.icbc.OperatorBC(geom, lambda X, Y, _: boundary_flux(X,Y)[8], boundary_left_right)

bc_BottomTop_gradient_22 = dde.icbc.OperatorBC(geom, lambda X, Y, _: boundary_flux(X,Y)[6], boundary_bottom_top)
bc_BottomTop_gradient_12 = dde.icbc.OperatorBC(geom, lambda X, Y, _: boundary_flux(X,Y)[7], boundary_bottom_top)


bcs =  [
          bc_LeftRight_traction_11,  bc_BottomTop_traction_22,  bc_All_traction_12,
          bc_LeftRight_charge_1,     bc_BottomTop_charge_2,
          bc_LeftRight_gradient_11,  bc_LeftRight_gradient_21,  bc_BottomTop_gradient_22,  bc_BottomTop_gradient_12,
         ]

data = dde.data.PDE(
    geom,
    pde,
    bcs,
    num_domain   = 10000,
    num_boundary = 1000,
    num_test     = 50000,
    train_distribution='uniform',
    anchors = None,
)

print((data.train_x_all).shape, (data.train_x_bc).shape, (data.train_x).shape)


## network
nn_layer_size = [2] + [20] * 3 + [5]  
activation  = "tanh"
initializer = "Glorot normal"

net = dde.nn.FNN(nn_layer_size, activation, initializer)

def transform_func(X, Y):
    x,y = X[:,0:1], X[:,1:2] 
    u1, u2, phi, P1, P2 = Y[:, 0:1], Y[:, 1:2], Y[:, 2:3], Y[:, 3:4], Y[:, 4:5]

    P1_new  =  P1 * 0.1  
    P2_new  =  P2 * 0.1 
    
    u1_new = u1 * 1e-3
    u2_new = u2 * 1e-3
    phi_new = phi * 1e-1
    
    
    return torch.cat((u1_new, u2_new, phi_new, P1_new, P2_new), dim =1)

net.apply_output_transform(transform_func)

model = dde.Model(data, net)

data_path = Path("./example_A/epoch")
data_path.mkdir(parents=True, exist_ok=True)

checkpointer = dde.callbacks.ModelCheckpoint(
    filepath = data_path, 
    verbose  = 1,
    save_better_only=True,  
    period=1000,
    monitor = 'train loss'
)


loss_weights = [
                1,     1,    100,    1,     1,               ## 5 pdes 
                1,                                           ## 1 energy
                10,    10,  10,                              ## 3 bcs of traction free
                10,    10,                                   ## 2 bcs of charge free
                10,    10,  10,   10                         ## 4 bcs of gradient free                          
                ] 

def energy_error(y_true, y_pred):
    return torch.exp(torch.mean(y_pred))

loss_type = ['MSE', 'MSE', 'MSE',  'MSE', 'MSE',  
             energy_error, 
             'MSE', 'MSE', 'MSE', 
             'MSE', 'MSE',
             'MSE', 'MSE', 'MSE','MSE']


###################################################################################
############################ train ############################
###################################################################################
begin_time = datetime.now()
print("Training starts from {}".format(begin_time))

model.compile("adam", lr=1e-3, loss = loss_type, loss_weights = loss_weights)
losshistory, train_state = model.train(iterations = 10000, display_every=1000, model_save_path = data_path, callbacks=[checkpointer])

dde.optimizers.config.set_LBFGS_options(
    maxcor=100,
    ftol=0,
    gtol=1e-08,
    maxiter=10000,
    maxfun=None,
    maxls=50,
)


model.compile("L-BFGS", loss = loss_type, loss_weights = loss_weights)
losshistory, train_state = model.train(display_every=1000, model_save_path = data_path, callbacks=[checkpointer])

end_time = datetime.now()
print("Training ends at {}".format(end_time))
print("Total time spent on training: {}".format(end_time - begin_time))

dde.saveplot(losshistory, train_state, issave=True, isplot= True, output_dir=data_path)

###################################################################################
############################ train ############################
###################################################################################



