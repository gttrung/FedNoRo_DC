# python version 3.7.1
# -*- coding: utf-8 -*-
import numpy as np
def separate_users(args, dict_users):
    
    if args.n_new_clients != 0:
      new_users = {}

      for _ in range(args.n_new_clients):
            
            position = len(dict_users)
            data = dict_users.pop(position-1)
            new_users[position-1] = data

      return dict_users, new_users
    
    else:
      return dict_users, {}

def merge_users(dict_users, new_users, args, stage = 1):
    
    if stage == 1:
        n_new_clients = np.round(args.n_new_clients*args.stage_ratio)
    elif stage == 2:
        n_new_clients = np.ceil(args.n_new_clients*(1-args.stage_ratio))
                            
    for _ in range(n_new_clients):
          
          position = len(dict_users)
          dict_users[position] = new_users[position]

    return dict_users
