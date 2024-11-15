class pnl:
    def __init__(self,data_obj):
        self.data=data_obj
    def calc_pnl(self,old_value,pnl_date,old_weights,new_weights,
                 t_cost=5, #Measured in BIPS
                 margin=0.1,
                 int_shorts=0.045/250, #Measured in daily interest
                 int_loan=0.055/250,
                 int_mm=0.05/250,
                 return_type='nec'
                 ):
        #pnl_date refers to the date for which the returns will be measured 
        # (the day after the optimization has been done)
        old_dol_pos=old_weights*old_value
        new_dol_pos=new_weights*old_value
        trans_costs=sum(abs(new_dol_pos-old_dol_pos))*0.0001*t_cost

        post_trade_val=old_value-trans_costs

        total_margin=margin*sum(abs(new_dol_pos))
        loan_interest=int_loan*((new_dol_pos[new_dol_pos>0].sum())-total_margin)

        total_shorts=-(new_dol_pos[new_dol_pos<0]).sum()
        short_interest=int_shorts*total_shorts

        resid_cash=post_trade_val-total_margin
        mm_interest=(resid_cash*int_mm)

        financing=-loan_interest+short_interest+mm_interest
        final_dol_pos=new_dol_pos*(1+self.data.load_data(pnl_date,dtype='returns'))
        pnl=sum(final_dol_pos-new_dol_pos)
        final_value=post_trade_val+financing+pnl
        final_weights=final_dol_pos/final_value
        if return_type=='nec':
            return final_value,final_weights
        elif return_type=='all':
            return final_value,final_weights,trans_costs,loan_interest,short_interest,mm_interest,pnl
