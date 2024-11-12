import tensorflow as tf
from pnl import pnl
from coefs import weight_matrix
import numpy as np



class optimizer:
    def __init__(self,data_obj,signal_dict,rev_dict,price_dict,filter_dict,coef_matrix):
        self.data=data_obj
        self.signal_dict=signal_dict
        self.rev_dict=rev_dict
        self.price_dict=price_dict
        self.filter_dict=filter_dict
        self.pnl=pnl(data_obj)
        self.coef_matrix=coef_matrix

    def optimize(self,c_w0,date_old,
                 lam_1, #Mean Reversion
                 lam_2, #Trade Costs
                 lam_3, #Volatility
                 t=None, #Initial Trades (store as tf Variable)
                 theta_1=0.2, #Cash
                 theta_2=0.2, #Beta
                 theta_3=3.8, #Min Total Inv
                 theta_4=4.2, #Max Total Inv,
                 n_epochs=1000, #Epochs of Optimizer
                 initial_learning_rate=1e-5
                 ):
        #Load all variables

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=25,
            decay_rate=0.9,
            staircase=True,
        )
        self.optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)

        c_sig=self.signal_dict[date_old][self.filter_dict[date_old]]
        t = tf.Variable(np.zeros(c_sig.shape[0]), dtype=np.float64) if t is None else t
        c_rev=self.rev_dict[date_old][self.filter_dict[date_old]]
        c_A=weight_matrix(self.price_dict[date_old],self.coef_matrix)[self.filter_dict[date_old]].T
        c_cov=self.data.load_data(date_old,dtype='cov')
        c_beta=self.data.load_data(date_old,dtype='beta')

        def obj_func(t):
            #OPTIMIZATION
            ret = tf.reduce_sum(t * c_sig)
            rev = tf.reduce_sum(t * c_rev)
            c_w=tf.matmul(c_A,tf.expand_dims(t, axis=1))

            tcosts = tf.reduce_sum(tf.abs(c_w-tf.expand_dims(c_w0,axis=1)))
            vol=tf.reduce_sum(tf.matmul(tf.matmul(c_w,c_cov,transpose_a=True),c_w))
            #PENALTIES
            calc_cash=tf.abs(tf.reduce_sum(tf.matmul(tf.expand_dims(np.ones(c_beta.shape), axis=1),c_w,transpose_a=True)))
            cash_penalty=(tf.maximum(theta_1,calc_cash)-theta_1)

            calc_beta=tf.abs(tf.reduce_sum(tf.matmul(tf.expand_dims(c_beta, axis=1),c_w,transpose_a=True)))
            beta_penalty=(tf.maximum(theta_2,calc_beta)-theta_2)

            total_weight=tf.reduce_sum(tf.matmul(tf.abs(c_w),np.ones(c_w.shape),transpose_a=True))
            weight_penalty=(tf.maximum(total_weight,theta_4)-theta_4)+(theta_3-tf.minimum(total_weight,theta_3))
            
            return (-(10*ret + lam_1*rev - lam_2*tcosts - lam_3*vol))+(1e6*(cash_penalty+beta_penalty+weight_penalty))

        def compute_gradients(t):
            with tf.GradientTape() as tape:
                loss = obj_func(t)
            return tape.gradient(loss, t)
        
        for epoch in range(n_epochs):
            grads = compute_gradients(t)
            self.optimizer.apply_gradients(zip([grads], [t]))
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Objective value: {obj_func(t).numpy()}")
        
        return t,c_A@t,c_beta,c_cov