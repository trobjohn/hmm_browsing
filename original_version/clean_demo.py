import pandas as pd

df = pd.read_csv('./data/hmm_demos_2023-10-03.csv', low_memory = False)

#path = '/scratch/trj2j/hmm/u_cat/'
path = './u_cat/'
aggregate_stats = []

counter = 0 

g_list = df['gender'].unique().tolist()
r_list = df['race_id'].unique().tolist()
eth_list = df['ethnicity_id'].unique().tolist()
edu_list = df['hoh_education'].unique().tolist()

for gen in g_list :    
    df_g = df[ (df['gender']==gen)]
    for race in r_list :
        df_r = df_g[ (df_g['race_id']==race) ]
        for eth in eth_list :
                df_eth = df_r[ (df_r['ethnicity_id']==eth) ]
                df_uc = df_eth 
                hits = df_uc.shape[0] 
                if hits > 0:
                    fn = path+'cat_'+str(counter)+'.csv'
                    df_uc.to_csv(fn)
                    aggregate_stats.append( [counter, df_uc.shape[0],gen,race,eth] )
                    counter += 1 

agg = pd.DataFrame(aggregate_stats,columns = ['counter','size','gen_code','race_code','eth_code'])
agg.sort_values('size',ascending=False,ignore_index=True,inplace=True)

agg.to_csv('./sweep/u_groups.csv',index=False)