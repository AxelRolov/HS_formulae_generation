from __future__ import division
import pandas as pd
import numpy as np
from numpy import inf
import math
import warnings
import matplotlib as mt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.mlab import bivariate_normal
import re
import glob
import sys
import seaborn as sns


######################################################Functions###############################################################

def tolist(row):
    row=row.tolist()
    return row

def vankrev_vis(df, flag=None):
    if(flag=='H/C, O/C' or flag==None):
        sns.mpl.rc("figure", figsize=(9,4))
        df=df[['H/C', 'O/C', 'Label']]
        ax=sns.lmplot(x="H/C", y="O/C", fit_reg=False, hue="Label", data=df)

def formdict(formlist):
    formulas_dict = {}
    for i in range(len(formlist)):
        formulas_dict[i] = formlist[i]
    return formulas_dict

def eucl_sim(heatmap):
    heatmap = heatmap[heatmap.columns].astype(float)
    heatmap=heatmap.div(heatmap.values.max())
    heatmap=1-heatmap
    return heatmap

def count_elrel(df):
       df['H/C']=df['H'].map(float)/df['C'].map(float)
       df['O/C']=df['O'].map(float)/df['C'].map(float)
       df['DBE']=(df['C'].map(float)+1)-(df['H'].map(float)/2)
       return df

def heatmap_vis(heatmap):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)        
    ax=sns.heatmap(heatmap, linewidth=.5,cmap='YlOrBr',square=True, vmin=heatmap.values.min(), vmax=1, annot=True)        
    ax.xaxis.tick_top()
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
    ax.set_yticklabels(ax.yaxis.get_majorticklabels(), rotation=0)
    plt.savefig("~/heatmap.svg")
    
def atom_count(df, elements=None):
    if (elements=='CHO' or elements==None):
        df['AtomCount']=df['C'].map(int)+df['H'].map(int)+df['O'].map(int)
    elif (elements=='CHONS'):
        df['AtomCount']=df['C'].map(int)+df['H'].map(int)+df['N'].map(int)+df['O'].map(int)+df['S'].map(int)
    return df

def formula(df, elements=None, flag=None):
    if (flag==None):
        if (elements=='CHO' or elements==None):
            df=df.loc[(df['N']==0) & (df['S']==0)]
            df['Formula']='C'+df['C'].map(str)+'H'+df['H'].map(str)+'O'+df['O'].map(str)
        elif (elements=='CHONS'):
            df['Formula']='C'+df['C'].map(str)+'H'+df['H'].map(str)+'N'+df['N'].map(str)+'O'+df['O'].map(str)+'S'+df['S'].map(str)
        df['Formula']=df['Formula'].str.replace('[C, H, O, N, S]0', '')
    elif(flag=='element'):
        df['C']=df['Formula'].str.extract('C(\d*)', expand=True)
        df['H']=df['Formula'].str.extract('H(\d*)', expand=True)
        df['O']=df['Formula'].str.extract('O(\d*)', expand=True)
    return df

def heatmap(fp, source_list, fp_type=None, dist_metric=None):
    heatmap=pd.DataFrame(index=source_list, columns=source_list)    
    if (fp_type=='bool' or fp_type==None):
        for i in source_list:
            vec=fp.loc[i]['Fingerprint']
            if(dist_metric=='Euclidean'):            
                for item, row in fp.iterrows():
                    heatmap.loc[i][item]=np.linalg.norm((vec-row['Fingerprint']))
            elif(dist_metric==None):
                print('Please, choose the metric for the distance measure')
            elif(dist_metric=='Tanimoto'):
                for item, row in fp.iterrows():
                    heatmap.loc[i][item]=float(float(np.sum(vec==row['Fingerprint']))/(len(vec)+len(row['Fingerprint'])-np.sum(vec==row['Fingerprint'])))    
                heatmap = heatmap[heatmap.columns].astype(float)
        if(dist_metric=='Euclidean'):
            heatmap=eucl_sim(heatmap)
        return heatmap
    elif(fp_type=='relative'):
        for i in source_list:
            vec=fp.loc[i]['Fingerprint']
            if(dist_metric=='Euclidean'):            
                for item, row in fp.iterrows():
                    heatmap.loc[i][item]=np.linalg.norm((vec-row['Fingerprint']))
            elif(dist_metric==None or dist_metric=='Tanimoto'):
                print('Please, choose the metric for the distance measure. You can\'t choose \'Tanimoto\' for the fp_type=relative')
        heatmap=eucl_sim(heatmap)
        return heatmap
    
def fingerprint(df, source_list, formulas_list, flag=None, space=None):   
    formulas_dict=formdict(formulas_list)    
    if (flag=='bool' or flag==None or flag=='form'):
        fp_bool=pd.DataFrame.from_records(formulas_dict, columns=formulas_list, index=source_list)
        for item in source_list:
            fp_bool.loc[str(item),list(df['Formula'][df['Source']==str(item)])]=1
        fp_bool=fp_bool.fillna(0)
        size=len(formulas_list)
        nums = np.zeros(size)
        nums[:2347] = 1
        np.random.shuffle(nums)
        nums=nums.astype(int)
        np.copyto(fp_bool.loc['Random'].T.values,nums)
        fp_bool['Fingerprint']=fp_bool[fp_bool.columns].values.tolist()###adding all values to one column
        fp_bool['Fingerprint']=fp_bool['Fingerprint'].map(lambda x: np.asarray(x))
        if (flag=='bool' or flag==None):
            return fp_bool
        elif(flag=='form'):
            fp_bool.drop('Random',inplace=True)
            fp_form=fp_bool.T
            fp_form=fp_form[~(fp_form.index=='Fingerprint')]
            fp_form['Formula']=fp_form.index
            if(space==None):
                mask=df[['DBE', 'class', 'C', 'H', 'O', 'N', 'S', 'O/C', 'H/C','DBE-Alex', 'Formula']]
                mask['Formula']=df['Formula'].drop_duplicates().dropna()
                mask=mask[mask['Formula'].notnull()]
                fp_form=pd.merge(fp_form, mask, how='inner', on='Formula')
            if(space=='generated'):
                fp_form=formula(fp_form, flag='element')
            fp_form=atom_count(fp_form)
            fp_form['IAC']=IAC(fp_form, ['C', 'H', 'O'])
            fp_form['MIAC']=MIAC(fp_form, ['C', 'H', 'O'])
            fp_form['Source']=np.nan
            for i, row in fp_form.iterrows():
                 if row[source_list].sum()==0:
                     fp_form.loc[i, 'Source']='Unpresented'
                 elif row[source_list].sum()==1:
                     fp_form.loc[i, 'Source']=str(row[row==1].index[0])
                 else:
                     #row=row[source_list]
                     #source_set=set(list(row[row.isin([1])].index))
                     fp_form.loc[i, 'Source']='Multi'#str(','.join(source_set))
            if(space==None):       
                fp_form.to_csv('/home/alex/Gumates_fp_form.csv', index=None, sep=',')
            elif(space=='Generated'):
                fp_form.to_csv('/home/alex/Gumates_fp_form_space.csv', index=None, sep=',')
            return fp_form
    elif (flag=='relative'):
        fp_rel=pd.DataFrame.from_records(formulas_dict, columns=formulas_list, index=source_list)    
        for item in source_list:
            fp_rel.loc[str(item),list(df['Formula'][df['Source']==str(item)])]=df['intensity'][df['Source']==str(item)].values
        fp_rel=fp_rel.fillna(0)
        np.copyto(fp_rel.loc['Random'].T.values,np.random.rand(1, (len(list(df['Formula'].unique())))))
        fp_rel['Fingerprint']=fp_rel[fp_rel.columns].values.tolist()###adding all values to one column
        fp_rel['Fingerprint']=fp_rel['Fingerprint'].map(lambda x: np.asarray(x))
        fp_rel['Fingerprint']=fp_rel['Fingerprint'].apply(lambda x: x/x.max())
        return fp_rel

def senior_rules(df):
   for element in ['C', 'H', 'O']:
       df[element] = df[element].astype(int)
   df['sumval'] = 4*df['C']+2*df['O']+1*df['H']
   df.loc[df['sumval']%2==0, 'rule1'] = 1
   df.loc[df['sumval']%2>0, 'rule1'] = 0
   df.loc[df['sumval']>=8, 'rule2'] = 1
   df.loc[df['sumval']<8, 'rule2'] = 0
   df.loc[df['sumval']>=2*(df['C']+df['H']+df['O']-1), 'rule3'] = 1
   df.loc[df['sumval']<2*(df['C']+df['H']+df['O']-1), 'rule3'] = 0
   return df    
    
def IAC(df, collist):
    shlist=[]    
    for item in collist:
        df[item]=df[item].fillna(0)
        x=np.log2(df[item].astype(float))
        x[x==-inf]=0
        shlist.append(df[item].astype(float)*x)
    return (df['AtomCount'].astype(float)*np.log2(df['AtomCount'].astype(float))-sum(shlist))

def MIAC(df, collist):
    shlist=[]    
    for item in collist:
        df[item]=df[item].fillna(0)
        x=np.log2(df[item].astype(float)/df['AtomCount'].astype(float))
        x[x==-inf]=0    
        #print x
        shlist.append((-1)*(df[item].astype(float)/(df['AtomCount'].astype(float)))*x)
    return sum(shlist)  

def count_combs(df, flag, left, i, comb, add):
    if (flag == None or flag == 'CHO'):
         aw = [12, 1, 16] ##list of carbon, hydrogen, oxygen atomic weights 
         names = {12: "C", 1: "H", 16: "O"} ##dictionary of aw-element symbols
    elif(flag == 'CHONS'):
         aw = [12, 1, 14, 16, 32,] ##list of carbon, hydrogen, oxygen atomic weights 
         names = {12: "C", 1: "H", 14: "N", 16 : "O", 32: "S"} ##dictionary of aw-element symbols
    if add: comb.append(add)
    if left == 0 or (i+1) == len(aw):
        if (i+1) == len(aw) and left > 0:
            if left % aw[i]: # can't get the exact score with this kind of aw
                return 0         # so give up on this recursive branch
            comb.append( (left/aw[i], aw[i]) ) # fix the amount here
            i += 1
        while i < len(aw):
            comb.append( (0, aw[i]) )
            i += 1
        df.append(str("".join("%s%d" % (names[c],n) for (n,c) in comb)))
        return 1
    cur = aw[i]
    return sum(count_combs(df, flag, left-x*cur, i+1, comb[:], (x,cur)) for x in range(0, int(left/cur)+1))

def space(mw, flag):
    df=[]
    full_list=[]
    if(flag=='CHO'):
        step=2
    elif(flag=='CHONS'):
        step=1
    for mw in range(mw[0], mw[1], step):
        count_combs(df, flag, mw, 0, [], None)
        full_list.append(df)
        df=[]
    flat_list=[item for sublist in full_list for item in sublist]
    return flat_list

def restrict(flat_list):
    rational_formulas_list=[x for x in flat_list if ((int(re.search(('C(\d*)'),x).group(1))>0) & (int(re.search(('H(\d*)'),x).group(1))>0) & (int(re.search(('O(\d*)'),x).group(1))>0))]
    gumate_formulas_list=[x for x in rational_formulas_list if ((float((float(re.search(('H(\d*)'),x).group(1)))/(float(re.search(('C(\d*)'),x).group(1))))>=0.27) & (float((float(re.search(('H(\d*)'),x).group(1)))/(float(re.search(('C(\d*)'),x).group(1))))<=2.2)&(float((float(re.search(('O(\d*)'),x).group(1)))/(float(re.search(('C(\d*)'),x).group(1))))>0)&(float((float(re.search(('O(\d*)'),x).group(1)))/(float(re.search(('C(\d*)'),x).group(1))))<=1))]
    return gumate_formulas_list
