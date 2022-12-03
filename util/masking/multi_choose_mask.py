import torch
import numpy as np
import random

def masking_mechanism(attentions, method, layer=11, power=0.5, high=True): # function for creating masks (based on attention) for each crop. The mask could be different for each crop.
    attn_m = attentions
    if method[:3] == 'raw':
        attn_cls_all_heads=attn_m[layer][:,:,0,1:] # picking only the cls attention for a specific given layer
        aver_of_all_heads=torch.mean(attn_cls_all_heads, dim=1) # calculate average over the heads for this layer
        masks = create_mask(aver_of_all_heads, high) # create the mask
    elif method[:3] == 'log':
        attn_cls_all_heads=attn_m[layer][:,:,0,1:]
        aver_of_all_heads=avg_log(attn_cls_all_heads)
        masks = create_mask(aver_of_all_heads, high)
    elif method[:3] == 'pow':
        attn_cls_all_heads=attn_m[layer][:,:,0,1:]
        aver_of_all_heads=avg_pow(attn_cls_all_heads, power)
        masks = create_mask(aver_of_all_heads, high)
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
        masks = create_mask(cls_A, high) # create the mask
    return(masks)


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
            add = add + attn_cls_all_heads[b, i, :] ** power 
        aver_attn = add/attn_cls_all_heads.shape[1]
        aver_attn_m.append(aver_attn)
    return(aver_attn_m)

def create_mask(aver_of_all_heads, high):
    aver_attn_m = aver_of_all_heads
    
    im_attentions_bl = []

    for j in range(0, len(aver_attn_m)):
        im1_attention_bl = torch.cuda.BoolTensor(196)

        val, idx = torch.sort(aver_attn_m[j],descending=high)
        r=np.random.uniform(0.1,0.5) # each time a random rate (10-50%) of the total number of patches per image will be masked.
        k=int(r*len(val))
        khigh_idx=idx[:k]
        for i in range(0, len(val)):
            if i in khigh_idx:

                im1_attention_bl[i]=1
            else:

                im1_attention_bl[i]=0
        k_prob = random.randint(0, 1) # each image will be masked or not with 50 % probability
        if k_prob == 0:
            im1_attention_bl=torch.zeros(196, dtype=torch.bool).cuda()

        im_attentions_bl.append(im1_attention_bl)

    tb = torch.stack(im_attentions_bl, dim=0)
    reshaped_cls = torch.reshape(tb, (len(aver_attn_m), 14, 14))

    return reshaped_cls