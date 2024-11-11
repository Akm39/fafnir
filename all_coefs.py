from grouptester import group_to_coef,agg_sectors
from tqdm.notebook import tqdm
from utils import read_pkl

class all_coefs:
    def __init__(self, data_obj):
        self.data=data_obj
        self.sector_list=self.data.sector_list()
    def run(self,t_stat=0.25,weight_imb=1,corr_thresh=0.9,weight_type='abs'):
        prog=tqdm(total=len(self.sector_list))
        self.coef_matrix=[]
        self.error_sector=[]
        self.size_error=[]
        for sector in self.sector_list:
            loaded_data=self.data.load_price(sector,tnt='train')
            try:
                if loaded_data.shape[1]>35:
                    max_its=4 #to limit comupute costs
                else:
                    max_its=5
                self.coef_matrix.append(group_to_coef(loaded_data,sector,max_its,t_stat,weight_imb,corr_thresh,weight_type))
            except:
                self.error_sector.append(sector)
                self.size_error.append(loaded_data.shape[1])
                print('Errors in: '+sector)
            prog.update(1)
    def errors(self):
        return dict(zip(self.error_sector,self.size_error))
    def coefs(self):
        full_matrix=agg_sectors(self.coef_matrix)
        all_companies=list(self.data.load_price().columns)
        companies_used=list(full_matrix.columns)
        for ticker in all_companies:
            if ticker not in companies_used:
                full_matrix[ticker]=0
        self.final=full_matrix[all_companies+['CONST']]
        return self.final
    def dump_matrix(self,name):
        root='data/'+self.data.dataset+'/'+name
        self.final.to_pickle(root)

def load_coefs(data_obj,coef_filename):
    path='data/'+data_obj.dataset+'/'+coef_filename
    return read_pkl(path)