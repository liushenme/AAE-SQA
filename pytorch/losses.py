import torch
import torch.nn.functional as F
import torch.nn as nn


def clip_bce(output_dict, target_dict):
    """Binary cross entropy loss.

    Args:
      output_dict: {'clipwise_output': (N, classes_num)}
      target_dict: {'target': (N, classes_num)}
    """
    return F.binary_cross_entropy(output_dict['clipwise_output'], target_dict['target'])

def clip_mse(output_dict, target_dict):
    """Binary cross entropy loss.

    Args:
      output_dict: {'clipwise_output': (N, classes_num)}
      target_dict: {'target': (N, classes_num)}
    """
    return F.mse_loss(output_dict['clipwise_output'], target_dict['target'])

def clip_frame_mse(output_dict, target_dict):
    """Binary cross entropy loss.

    Args:
      output_dict: {'clipwise_output': (N, classes_num)}
      target_dict: {'target': (N, classes_num)}
    """
    clip_out = output_dict['clipwise_output']
    frame_out = output_dict['framewise_output']     
    target = target_dict['target']
    clip_mse = torch.mean((clip_out - target) ** 2)
    frame_mse = torch.mean((1 * torch.mean((frame_out - target) ** 2, dim=1, keepdim=True))
    loss = clip_mse + frame_mse
    return loss

#def frame_mse(y_true, y_pred):  # Customized loss function  (frame-level loss, the second term of equation 1 in the paper)
#    True_pesq=y_true[0,0]           
#        return (10**(True_pesq-4.5))*tf.reduce_mean((y_true-y_pred)**2)

def get_loss_func(loss_type):
    if loss_type == 'clip_bce':
        return clip_bce
    elif loss_type == 'clip_mse':
        return clip_mse
    elif loss_type == 'clip_frame_mse':
        return clip_frame_mse



if __name__ == '__main__':
    output_dict = torch.tensor([[3],
                                [4]])
    output_dict = output_dict.float()
    output_dict_frame = torch.tensor([[2.5,4],
                                      [4,3.5]])
    output_dict_frame = output_dict_frame.float()
    target_dict = torch.tensor([[2],
                                [3]])
    target_dict = target_dict.float()
    clip_mse = torch.mean((output_dict - target_dict) ** 2)
    print(clip_mse)

    frame_mse = (10 ** (target_dict - 5)) * torch.mean((output_dict_frame - target_dict) ** 2, dim =1, keepdim=True)
    print((10 ** (target_dict - 5)))
    print(torch.mean((output_dict_frame - target_dict) ** 2, dim =1, keepdim=True))
    print(frame_mse)
    loss = clip_mse + frame_mse
    



