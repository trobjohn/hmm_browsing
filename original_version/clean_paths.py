import os as os
import gzip as gz
import pandas as pd
from multiprocessing import Pool

# lscpu
# 10 seconds per file, 2327 files... 23,270 seconds/ X cores



def process_file(file):
    global files, completed
    """ Function to process users and create their path/target files. """
    if file.endswith(".csv.gz") and file not in completed:
        #print('Processing ', file)
        nat = pd.Series(['']) # pd.Series([np.nan]) # pd.Series([''])
        f = gz.open('./t_hmm_2023-10-03/'+file)
        df = pd.read_csv(f,low_memory=False)
        df = df.drop(['tdate','category','subcategory','refer_domain'],axis=1)
        df = df.rename({'timestamp_ss2k':'time'},axis=1)
        df['from'] = pd.concat([nat,df.loc[0:df.shape[0],'domain']],axis=0,ignore_index=True)
        df['to'] = pd.concat([df.loc[1:(df.shape[0]-1),'domain'],nat],axis=0,ignore_index=True)
        df['dt'] = df.loc[:,'time'].diff()
        users = df['person_id'].unique()
        for u in users:
            df_u = df.loc[ df['person_id']==u,: ]
            df_u = df_u.drop('person_id',axis=1)
            u_set = set( df_u['domain'].unique() ) # All sites visited
            target_visits = target_set.intersection(u_set) # Intersection with target set
            if len(target_visits) > 0:
                rolo = open("./sweep/rolodex_u.csv", "a")
                rolo.write(str(u)+'\n')
                rolo.close()
                if os.path.exists('path_'+str(u)+'.csv'):
                    #print('Appended, ', u)
                    df_u.to_csv(rivanna_path+'u_paths/path_'+str(u)+'.csv',mode='a',header=False,index=False)
                else:
                    #print('New user, ', u)
                    df_u.to_csv(rivanna_path+'u_paths/path_'+str(u)+'.csv',index=False)
        print('Processed', file)
        compl = open("./sweep/completed.csv", "a")
        compl.write(file+'\n')
        compl.close()  





if __name__ == '__main__':

    POOL_SIZE = 24 # 16-32 locally
    #rivanna_path = './'
    rivanna_path = '/scratch/trj2j/hmm/'
    #rivanna_path = '/media/trj/Extreme Pro/hmm/'

    # Target domains:
    target = pd.read_csv('./data/news.csv')
    target_set = set( target['domain'])

    # Create user and site rolodexes
    rolo = open("./sweep/rolodex_u.csv", "w")
    rolo.write('user\n')
    rolo.close()

    comp = open("./sweep/completed.csv", "w")
    comp.write('file\n')
    comp.close()
    completed = pd.read_csv('./sweep/completed.csv')

    ## Process .csv.gz files:
    files = os.listdir('./t_hmm_2023-10-03/')
    POOL = Pool(POOL_SIZE)
    POOL.map(process_file,files)
    POOL.close()
    POOL.join()
    #finished = pd.to_csv('./sweep/completed.csv')