'''
Read Sepsis challenge data for data imputation 
@author: Srinivasan Sivanandan

Adapted from GRUI implementation by @author: lyh
'''

import sys
sys.path.append("..")
import os
import random
import math
import numpy as np 

class ReadSepsisData():
    # first read all dataset
    # before call, determine wheher shuffle
    # produce next batch
    def __init__(self, dataPath,isNormal):
        self.dataPath = dataPath
        print("data path: "+self.dataPath)
        self.fileNames = os.listdir(self.dataPath)
        labels=[]
        
        self.dic = {'HR':0, 'O2Sat':1, 'Temp':2,
                    'SBP':3,'MAP':4, 'DBP':5,
                    'Resp':6, 'EtCO2':7, 'BaseExcess':8,
                    'HCO3':9, 'FiO2':10, 'pH':11, 'PaCO2':12,
                    'SaO2':13, 'AST':14, 'BUN':15, 'Alkalinephos':16,
                    'Calcium':17, 'Chloride':18, 'Creatinine':19,
                    'Bilirubin_direct':20, 'Glucose':21, 'Lactate':22, 
                    'Magnesium':23, 'Phosphate':24, 'Potassium':25, 'Bilirubin_total':26,
                    'TroponinI':27, 'Hct':28, 'Hgb':29, 'PTT':30, 
                    'WBC':31, 'Fibrinogen':32, 'Platelets':33, 'Age':34, 
                    'Gender':35, 'SepsisLabel':40, 'ICULOS':39}
        # List to store the sum and count of values
        self.nFeatures = len(self.dic)-2 
        mean=[0.0]*(self.nFeatures)
        meancount=[0]*(self.nFeatures)
        x=[]
        labels = []
        times=[]
        non_in_dic_count=0
        # times: totalFilesLength*steps
        # x: totalFilesLength*steps*feature_length
        for fileName in self.fileNames:
            f=open(os.path.join(self.dataPath, fileName))
            count=0
            totalData=[]
            t_times=[]
            sepsis_label = []
            for line in f.readlines():
                if count > 0:
                    data = [float(pt) for pt in line.strip().split('|')]
                    for i in range(0, self.nFeatures):
                        if not np.isnan(data[i]) and data[i]!=0:
                            mean[i]+=data[i]
                            meancount[i]+=1
                        #Convert nans to 0 for processing
                        if np.isnan(data[i]):
                            data[i] = 0.0
                    # Assuming the features are the first set of values in the dataset ** IMPORTANT **
                    totalData.append(data[0:self.nFeatures]) 
                    t_times.append(float(data[self.dic['ICULOS']]))
                    label = [0,1] if float(data[self.dic['SepsisLabel']])==1 else [1,0]
                    sepsis_label.append(label)           
                count+=1            
            x.append(totalData)
            times.append(t_times)
            labels.append(sepsis_label)
            f.close()
       
        self.x=x
        self.y=labels
        self.times=times        
        
        for i in range(len(mean)):
            if meancount[i]!=0:
                mean[i]=mean[i]/meancount[i]
        self.mean=mean
        
        
        # normalization
        m=[] # mask 0/1
        # first calculate std
        self.std=[0.0]*(self.nFeatures)
        for onefile in self.x:
            one_m=[]
            for oneclass in onefile:
                t_m=[0]*len(oneclass)
                for j in range(len(oneclass)):
                    if not np.isnan(oneclass[j]) and oneclass[j]!=0:
                        self.std[j]+=(oneclass[j]-self.mean[j])**2
                        t_m[j]=1
                one_m.append(t_m)
            m.append(one_m)
        for j in range(len(self.std)):
            if meancount[j]>1:
                self.std[j]=math.sqrt(1.0/(meancount[j]-1)*self.std[j])

        self.isNormal=isNormal
        self.normalization(isNormal)    
                      
        x_lengths=[] #
        deltaPre=[] #time difference 
        lastvalues=[] # if missing, last values
        deltaSub=[]
        subvalues=[]
        for h in range(len(self.x)):
            # oneFile: steps*value_number
            oneFile=self.x[h]
            one_time=self.times[h]
            x_lengths.append(len(oneFile))
            
            one_deltaPre=[]
            one_lastvalues=[]
            
            one_deltaSub=[]
            one_subvalues=[]
            
            one_m=m[h]
            for i in range(len(oneFile)):
                t_deltaPre=[0.0]*len(oneFile[i])
                t_lastvalue=[0.0]*len(oneFile[i])
                one_deltaPre.append(t_deltaPre)
                one_lastvalues.append(t_lastvalue)
                
                if i==0:
                    for j in range(len(oneFile[i])):
                        one_lastvalues[i][j]=0.0 if one_m[i][j]==0 else oneFile[i][j]
                    continue
                for j in range(len(oneFile[i])):
                    if one_m[i-1][j]==1:
                        one_deltaPre[i][j]=one_time[i]-one_time[i-1]
                    if one_m[i-1][j]==0:
                        one_deltaPre[i][j]=one_time[i]-one_time[i-1]+one_deltaPre[i-1][j]
                        
                    if one_m[i][j]==1:
                        one_lastvalues[i][j]=oneFile[i][j]
                    if one_m[i][j]==0:
                        one_lastvalues[i][j]=one_lastvalues[i-1][j]
        
            for i in range(len(oneFile)):
                t_deltaSub=[0.0]*len(oneFile[i])
                t_subvalue=[0.0]*len(oneFile[i])
                one_deltaSub.append(t_deltaSub)
                one_subvalues.append(t_subvalue)
            #construct array 
            for i in range(len(oneFile)-1,-1,-1):    
                if i==len(oneFile)-1:
                    for j in range(len(oneFile[i])):
                        one_subvalues[i][j]=0.0 if one_m[i][j]==0 else oneFile[i][j]
                    continue
                for j in range(len(oneFile[i])):
                    if one_m[i+1][j]==1:
                        one_deltaSub[i][j]=one_time[i+1]-one_time[i]
                    if one_m[i+1][j]==0:
                        one_deltaSub[i][j]=one_time[i+1]-one_time[i]+one_deltaSub[i+1][j]
                        
                    if one_m[i][j]==1:
                        one_subvalues[i][j]=oneFile[i][j]
                    if one_m[i][j]==0:
                        one_subvalues[i][j]=one_subvalues[i+1][j]   
                
            
            #m.append(one_m)
            deltaPre.append(one_deltaPre)
            lastvalues.append(one_lastvalues)
            deltaSub.append(one_deltaSub)
            subvalues.append(one_subvalues)
        self.m=m
        self.deltaPre=deltaPre
        self.lastvalues=lastvalues
        self.deltaSub=deltaSub
        self.subvalues=subvalues
        self.x_lengths=x_lengths
        self.maxLength=max(x_lengths)
        
        print("max_length is : "+str(self.maxLength))
        print("non_in_dic_count is : "+str(non_in_dic_count))
        
        resultFile=open(os.path.join("./","meanAndstd"),'w')
        for i in range(len(self.mean)):
            resultFile.writelines(str(self.mean[i])+","+str(self.std[i])+","+str(meancount[i])+"\r\n")
        resultFile.close()
        
    def normalization(self,isNormal):
        if not isNormal:
            return
        for onefile in self.x:
            for oneclass in onefile:
                for j in range(len(oneclass)):
                    if oneclass[j] !=0:
                        if self.std[j]==0:
                            oneclass[j]=oneclass[j] # Do not normalize if STD = 0
                        else:
                            oneclass[j]=1.0/self.std[j]*(oneclass[j]-self.mean[j])
    
    def nextBatch(self):
        i=1
        while i*self.batchSize<=len(self.x):
            x=[]
            y=[]
            m=[]
            deltaPre=[]
            x_lengths=[]
            lastvalues=[]
            deltaSub=[]
            subvalues=[]
            imputed_deltapre=[]
            imputed_m=[]
            imputed_deltasub=[]
            mean=self.mean
            files=[]
            for j in range((i-1)*self.batchSize,i*self.batchSize):
                files.append(self.fileNames[j])
                x.append(self.x[j])
                y.append(self.y[j])
                m.append(self.m[j])
                
                deltaPre.append(self.deltaPre[j])
                deltaSub.append(self.deltaSub[j])
                #放的都是引用，下面添加0，则原始数据也加了0
                x_lengths.append(self.x_lengths[j])
                lastvalues.append(self.lastvalues[j])
                subvalues.append(self.subvalues[j])
                jj=j-(i-1)*self.batchSize
                #times.append(self.times[j])
                while len(x[jj])<self.maxLength:
                    t1=[0.0]*(self.nFeatures)
                    x[jj].append(t1)
                    y[jj].append(y[jj][len(y[jj])-1])
                    t2=[0]*(self.nFeatures)
                    m[jj].append(t2)
                    t3=[0.0]*(self.nFeatures)
                    deltaPre[jj].append(t3)
                    t4=[0.0]*(self.nFeatures)
                    lastvalues[jj].append(t4)
                    t5=[0.0]*(self.nFeatures)
                    deltaSub[jj].append(t5)
                    t6=[0.0]*(self.nFeatures)
                    subvalues[jj].append(t6)
            for j in range((i-1)*self.batchSize,i*self.batchSize):
                one_imputed_deltapre=[]
                one_imputed_deltasub=[]
                one_G_m=[]
                for h in range(0,self.x_lengths[j]):
                    if h==0:
                        one_f_time=[0.0]*(self.nFeatures)
                        one_imputed_deltapre.append(one_f_time)
                        try:
                            one_sub=[self.times[j][h+1]-self.times[j][h]]*(self.nFeatures)
                        except:
                            print("error: "+str(h)+" "+str(len(self.times[j]))+" "+self.fileNames[j])
                        one_imputed_deltasub.append(one_sub)
                        one_f_g_m=[1.0]*(self.nFeatures)
                        one_G_m.append(one_f_g_m)
                    elif h==self.x_lengths[j]-1:
                        one_f_time=[self.times[j][h]-self.times[j][h-1]]*(self.nFeatures)
                        one_imputed_deltapre.append(one_f_time)
                        one_sub=[0.0]*(self.nFeatures)
                        one_imputed_deltasub.append(one_sub)
                        one_f_g_m=[1.0]*(self.nFeatures)
                        one_G_m.append(one_f_g_m)
                    else:
                        one_f_time=[self.times[j][h]-self.times[j][h-1]]*(self.nFeatures)
                        one_imputed_deltapre.append(one_f_time)
                        one_sub=[self.times[j][h+1]-self.times[j][h]]*(self.nFeatures)
                        one_imputed_deltasub.append(one_sub)
                        one_f_g_m=[1.0]*(self.nFeatures)
                        one_G_m.append(one_f_g_m)
                while len(one_imputed_deltapre)<self.maxLength:
                    one_f_time=[0.0]*(self.nFeatures)
                    one_imputed_deltapre.append(one_f_time)
                    one_sub=[0.0]*(self.nFeatures)
                    one_imputed_deltasub.append(one_sub)
                    one_f_g_m=[0.0]*(self.nFeatures)
                    one_G_m.append(one_f_g_m)
                imputed_deltapre.append(one_imputed_deltapre)
                imputed_deltasub.append(one_imputed_deltasub)
                imputed_m.append(one_G_m)
                #重新设置times,times和delta类似，但times生成的时候m全是1,用于生成器G
            i+=1
            if self.isNormal:
                yield  x,y,[0.0]*(self.nFeatures),m,deltaPre,x_lengths,lastvalues,files,imputed_deltapre,imputed_m,deltaSub,subvalues,imputed_deltasub
            else:
                yield  x,y,mean,m,deltaPre,x_lengths,lastvalues,files,imputed_deltapre,imputed_m,deltaSub,subvalues,imputed_deltasub
                
    # Shuffle dataset
    def shuffle(self,batchSize=32,isShuffle=False):
        self.batchSize=batchSize
        if isShuffle:
            c = list(zip(self.x,self.y,self.m,self.deltaPre,self.x_lengths,self.lastvalues,self.fileNames,self.times,self.deltaSub,self.subvalues))
            random.shuffle(c)
            self.x,self.y,self.m,self.deltaPre,self.x_lengths,self.lastvalues,self.fileNames,self.times,self.deltaSub,self.subvalues=zip(*c)

if __name__ == '__main__':
    # Read Sepsis dataset & shuffle
    dt = ReadSepsisData("../../data/train" ,isNormal=True)
    dt.shuffle(1,False)
    batchCount=1
    X_lengths=dt.x_lengths
    print(sum(X_lengths)/len(X_lengths))

    # Read each batch and print files of every 100th batch
    for x,y,mean,m,deltaPre,x_lengths,lastvalues,files,imputed_deltapre,imputed_m,deltaSub,subvalues,imputed_deltasub in dt.nextBatch():
        print(batchCount)
        batchCount+=1
        if batchCount%1==0:
            print(y)

def f():
    print("read_sepsis_data")