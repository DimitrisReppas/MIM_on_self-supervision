import torch
import numpy as np
import random

def positive_negative_selection(attentions, k, batch_per_gpu, global_crops, method, layer=11, power=0.5, high=True): # function used to select positive and negative pairs.The selection mechanish is based on the attention map of each image
    attn_m = attentions
    
    if method[:3] == 'raw':
        attn_cls_all_heads=attn_m[layer][:,:,0,1:] # picking only the cls attention for a specific given layer
        aver_of_all_heads=torch.mean(attn_cls_all_heads, dim=1) # calculate average over the heads for this layer
        pos_neg = k_pos_neg(aver_of_all_heads, batch_per_gpu, global_crops, high, k) # create the mask
    elif method[:3] == 'log':
        attn_cls_all_heads=attn_m[layer][:,:,0,1:]
        aver_of_all_heads=avg_log(attn_cls_all_heads)
        pos_neg = k_pos_neg(aver_of_all_heads, batch_per_gpu, global_crops, high, k)
    elif method[:3] == 'pow':
        attn_cls_all_heads=attn_m[layer][:,:,0,1:]
        aver_of_all_heads=avg_pow(attn_cls_all_heads, power)
        pos_neg = k_pos_neg(aver_of_all_heads, batch_per_gpu, global_crops, high, k)
    else: #create mask when using rollout method
        attn_m_torch = torch.stack(attn_m, dim=0)
        aver_of_all_heads = torch.mean(attn_m_torch, dim=2) # calculate average over the heads for all layer
        avg_plus_I = aver_of_all_heads + 1.0*torch.eye(197).cuda() # take into account residual connections by adding the identity matrix
        
        if method[-4:] == "all_": # take into account all layer
            multip = avg_plus_I[0] @ avg_plus_I[1] @ avg_plus_I[2] @ avg_plus_I[3] @ avg_plus_I[4] @ avg_plus_I[5] @ avg_plus_I[6] @ avg_plus_I[7] @ avg_plus_I[8] @ avg_plus_I[9] @ avg_plus_I[10] @ avg_plus_I[11]
        elif method[-4:] == "last": # take into account part of the last layers
            multip = avg_plus_I[11] @ avg_plus_I[10] @ avg_plus_I[9] @ avg_plus_I[8]
        elif method[-4:] == "mid_": # take into account part of the middle layers
            multip = avg_plus_I[4] @ avg_plus_I[5] @ avg_plus_I[6] @ avg_plus_I[7]

        cls_A = multip[:,:1,1:] # picking only the cls attention
        cls_A = torch.squeeze(cls_A, 1)
        pos_neg = k_pos_neg(cls_A, batch_per_gpu, global_crops, high, k) # pick the k positives and k negatives based on the attention mask
    return(pos_neg)

def avg_log(attn_cls_all_heads):
    
    add = 0
    aver_attn_m = []
    for b in range(0,attn_cls_all_heads.shape[0]):
        for i in range(0,attn_cls_all_heads.shape[1]):
            add = add + torch.log(attn_cls_all_heads[b, i, :])
        aver_attn = add/attn_cls_all_heads.shape[1]
        aver_attn_m.append(aver_attn)
    return(aver_attn_m)
def avg_pow(attn_cls_all_heads, power):

    add = 0
    aver_attn_m = []
    for b in range(0,attn_cls_all_heads.shape[0]):
        for i in range(0,attn_cls_all_heads.shape[1]):
            add = add + attn_cls_all_heads[b, i, :] ** power # Change the parameter for different masking
        aver_attn = add/attn_cls_all_heads.shape[1]
        aver_attn_m.append(aver_attn)
    return(aver_attn_m)

def k_pos_neg(aver_of_all_heads, batch_per_gpu, global_crops, high, k):
    aver_attn_m = aver_of_all_heads
    khigh = torch.zeros(len(aver_attn_m),k)
    klow = torch.zeros(len(aver_attn_m),k)
    for j in range(0, len(aver_attn_m)):
        val, idx = torch.sort(aver_attn_m[j],descending=high)
        klow[j,:] = idx[-k:] # positives
        khigh[j,:] = idx[:k] # negatives

    return(khigh, klow)
