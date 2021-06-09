import numpy as np
import torch

def main():
    cared_tensor = torch.tensor([3, 8, 6, 8, 6, 6], dtype=float, requires_grad=True).cuda()
    cared_list = [val for val in cared_tensor]
    print("cared_tensor: {} \ncared_list: {}".format(cared_tensor, cared_list))
    print("cared_tensor.dtype: {}".format(cared_tensor.dtype))
    cnt_ = torch.sum(cared_tensor == 6)
    cnt_list = cared_list.count(6)
    print("cnt_: {}\ncnt_list: {}".format(cnt_, cnt_list))
    indicator = cared_tensor == 6
    print("indicator: {}".format(indicator))
    zero_tensor = torch.zeros(cared_tensor.size(), dtype=cared_tensor.dtype).cuda()
    print("zero_tensor.dtype: {}".format(zero_tensor.dtype))
    filtered_tensor = torch.where(indicator, cared_tensor, zero_tensor)
    print("filtered_tensor: {}".format(filtered_tensor))
    print("filtered_tensor.device: {}".format(filtered_tensor.device))
    dummy1 = torch.zeros(2)
    dummy2 = torch.zeros(2)
    dummy_prod = torch.dot(dummy1, dummy2)
    bool1 = dummy_prod > 1
    print("dummy_prod: {}\ntype(dummy_prod): {}\nbool1: {}".format(dummy_prod, type(dummy_prod), bool1))
    cared_tensor[0] = torch.tensor(float('nan')).cuda()
    valid_ind = ~torch.isnan(cared_tensor)
    print("valid_ind: {}".format(valid_ind))
    new_tensor = torch.where(valid_ind, cared_tensor, zero_tensor)
    print("new_tensor: {}".format(new_tensor))
    new_sum = torch.sum(new_tensor)
    print("new_sum: {}".format(new_sum))
    new_arr = new_tensor.detach().cpu().numpy()
    print("new_tensor: {}".format(new_tensor))
    print("new_tensor.requires_grad: {}".format(new_tensor.requires_grad))
    print("new_arr: {}".format(new_arr))

if __name__ == '__main__':
    main()