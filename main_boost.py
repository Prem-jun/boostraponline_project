#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 19:44:26 2024

@author: premjunsawang
"""
from boostraponline import stat_dist, managefile, boostrap_v1, booststream 
import random, pickle
import matplotlib.pyplot as plt
import math, copy
import numpy as np
import pandas as pd
from tabulate import tabulate
import typing
import os, glob
# Using the @classmethod Decorator
class Book:
   total_pages = 0
   def __init__(self, title, author, pages):
       self.title = title
       self.author = author
       self.pages = pages
   @classmethod
   def update_total_pages(cls, pages):
       cls.total_pages += pages
   
   def add_to_total_pages(self):
       Book.update_total_pages(self.pages)


    
    
# def main(self,filewd:str = None,filename: str = None) -> None:    
def main(self,list_sch: list, filename: str, pop_min: float, pop_max: float,\
         ch_size: int, feed_size: int) -> None:        
    # filename_full = filename+'.p'
    # [list_sch,percent_feed,pop_stat,samp_stat] = pickle.load(open(os.path.join(filewd, filename_full),"rb"))    
    # if self.filename =='' or self.filewd =='': # empty object
    # for i in range(len(list_sch)):
        
    
    if self.filename =='' or self.filewd =='':
        self.update_cls_att(filename, pop_min, pop_max, ch_size, feed_size)
        # filename_full = filename+'.p'
        # [list_sch,percent_feed,pop_stat,samp_stat] = pickle.load(open(os.path.join(filewd, filename_full),"rb"))    
        #for i in range(len(list_sch)):
            # ch_size = len(list_sch[i][0])
            # feed_size = len(list_sch[i])
            # pop_max = pop_stat['max']
            # pop_min = pop_stat['min']
            # self.update_cls_att(filename, pop_min, pop_max, ch_size, feed_size)
        # [dist,randomlist,feed_num,feed_ch,percent_feed,pop_max,pop_min,\
        #                       samp_max,samp_min] = pickle.load(open(os.path.join(filewd, filename_full),"rb"))    
        # ch_size = len(feed_ch[0])
        # feed_size = feed_num
        # self.update_cls_att(filewd, filename, pop_min, pop_max, ch_size, feed_size)
    # else:
    #     filename_full = self.filename+'.p'
    #     [dist,randomlist,feed_num,feed_ch,percent_feed,pop_max,pop_min,\
    #                           samp_max,samp_min] = pickle.load(open(os.path.join(self.filewd, filename_full),"rb"))    
    if addnoise is True:
       list_sch[7].append(pop_max+10)
       list_sch[5].append(pop_min-10)
    
    if self.online is False:
        # flatten list
        samp_wh = []
        for i in list_sch:
            samp_wh += i
        self.expand_whole(samp_wh)
        
    else:
        expansion = self.expand_init(list_sch[0])
        if expansion is True:
            self.expandch.append(1)
        print("=============================================================")
        print("\n--------- get next chunk")
        num_exp_chunk = len(list_sch)
        max_chs = max(list_sch[0])
        min_chs = min(list_sch[0])
        for i in range(1,num_exp_chunk):
            max_ch = max(list_sch[i])
            min_ch = min(list_sch[i])
            if max_ch > max_chs:
                max_chs = max_ch
            if min_ch < min_chs:
                min_chs = min_ch
            # range_chs = max_chs - min_chs    
            expansion  = self.expand(list_sch[i])
            if expansion is True:
                self.expandch.append(i) 
            self.update_range()
            
if __name__=='__main__':
    # data_no = 7
    addnoise = False
    wd = "./sim_data/"
    # folder_dist = "wald"
    folder_dist ='wiebull'
    file_path = os.path.join(wd, folder_dist,'Chunk')
    res_path = os.path.join(wd, folder_dist,'Chunk_res')
    list_ = glob.glob(os.path.join(file_path,'*.p'))
    if not os.path.isdir(res_path):
        os.makedirs(res_path)
    for file in list_:
        [filename,filetype] =os.path.basename(file).split('/')[-1].rsplit('.',1)
        print(f'Running: {filename}')
        [pop_data,list_sch,percent_feed,pop_stat,samp_stat] = pickle.load(open(file,"rb"))    
        # [dist,randomlist,feed_num,feed_ch,percent_feed,pop_max,pop_min,\
        #                      samp_max,samp_min] = pickle.load(open(file,"rb"))  
        list_bt_wh = [] # Whole boostrap
        list_bt_on = [] # Online boostrap
        list_bt_onmm = [] # Online minmax boostrap
        pop_max = pop_stat['max']
        pop_min = pop_stat['min']
        data_dict = {'file_run':filename,
                    'pop_data':pop_data,
                    'pop_stat':pop_stat,
                    'samp_stat':samp_stat
                    }
        for i in range(len(list_sch)):
            ch_size = len(list_sch[i][0])
            feed_size = len(list_sch[i])
            # Whole dataset method start
            list_bt_wh.append(booststream(online = False))
            main(list_bt_wh[i],list_sch[i],file, pop_min, pop_max, ch_size, feed_size)
            
            # online dataset method start
            list_bt_on.append(booststream())
            main(list_bt_on[i],list_sch[i],file, pop_min, pop_max, ch_size, feed_size)
            
            
            # minmax online data method start
            list_bt_onmm.append(booststream(minmax_boost=True))
            main(list_bt_onmm[i],list_sch[i],file, pop_min, pop_max, ch_size, feed_size)
        
        res_fullname ='res_'+filename+'.pickle'   
        with open(os.path.join(res_path, res_fullname), "wb") as f:
              pickle.dump([data_dict,list_bt_wh,list_bt_on,list_bt_onmm],f)
        
        
        # list_bt = [booststream(online = False), 
        #            booststream()
        #            ,booststream(minmax_boost=True),
        #         booststream(online = False,prob = True),
        #         booststream(prob=True),
        #         booststream(online = False,prob = True,minmax_boost=True),
        #         booststream(prob=True,minmax_boost=True)]
        
        # for i in range(len(list_bt)):
        #     main(list_bt[i],file_path,filename)
        # res_fullname ='res_'+filename+'.pickle'   
        # with open(os.path.join(res_path, res_fullname), "wb") as f:
        #      pickle.dump(list_bt,f)
       
        
       
        
       # pickle.dump([list_bt], open(os.path.join(res_path, res_fullname), "w"))    
        
    # dict_cond = {'online':[True,True,True,True,False,False],\
    #               'prob':[False,False,True,True,False,True],\
    #                   'minmax_boost':[False,True,False,True,False,False]}
    # cond_test = pd.DataFrame(dict_cond)    
    # # if data_no == 7:    
    # #     filename = 'wiebullshape5n20000-2ch-200-0'
    # ncon = cond_test.shape[0]
    # list_bt = []
    # for i in range(ncon):
    #     list_bt.append(booststream(cond_test.iloc[i,:]))
        
    #     if i == 0:
    #         main(list_bt[0],filewd,filename)
    #     else:
    #         main(list_bt[i])
        
    # for i in range(ncon):
    #     list_bt[i].report()
   
    
    
    # list_bt = [booststream(online = False), booststream(),booststream(minmax_boost=True),
    #            booststream(online = False,prob = True),booststream(prob=True),
    #            booststream(online = False,prob = True,minmax_boost=True),
    #            booststream(prob=True,minmax_boost=True)]
    # for i in range(len(list_bt)):
    #     if i == 0 :
    #         main(list_bt[i],filewd,filename)
    #     else:
    #         main(list_bt[i])
    # if addnoise is True:
    #     filename = filename+'Rnoise'        
    # else:    
    #     filename = filename+'R'        
    # for i in range(len(list_bt)):
    #     if i ==0:
    #         list_bt[i].report_xcel(filename)
    #     else:
    #         list_bt[i].report_xcel(filename,append = True)
    # pickle.dump([list_bt], open(filename+".p", "wb"))    
    
    
    # bt = booststream()
    # bt_p = booststream(prob=True)
    # bt_mm = booststream(minmax_boost=True)
    # bt_mm_p = booststream(prob=True,minmax_boost=True)
    # bt_w = booststream(online = False)
    # bt_w_prob = booststream(online = False,prob = True)    
    # for i in range(nround):
    #     # main(bt_w,filewd,filename)
    #     main(bt,filewd,filename)
    #     main(bt_p)
    #     main(bt_mm)
    #     main(bt_mm_p)
    #     main(bt_w)
    #     main(bt_w_prob)
        
        
    #     bt.report()
    #     bt_p.report()
    #     bt_mm.report()
    #     bt_mm_p.report()
    #     bt_w.report()
    #     bt_w_prob.report()
    # list_bt = [bt,bt_p,bt_mm,bt_mm_p,bt_w,bt_w_prob]
    # # pop_stat, samp_stat,list_expand, list_error, expand_chunk_list, list_bin_mean = main(filename=filename,addnoise=addnoise,minmax_boost=minmax_boost)    