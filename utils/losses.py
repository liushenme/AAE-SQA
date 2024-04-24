import torch
import torch.nn.functional as F
import torch.nn as nn
from audtorch.metrics.functional import pearsonr

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

def AWL(params, *x):
    loss_sum = 0
    for i, loss in enumerate(x):
        loss_sum += 0.5 / (params[i] ** 2) * loss + torch.log(1 + params[i] ** 2)
    return loss_sum

def mse_ce_3t_noae_awl(pred_score, target_score, pred_spk, target_spk, pred_cla, target_cla, params):
 
    score_loss = F.mse_loss(pred_score, target_score)
    spk_loss = F.cross_entropy(pred_spk, target_spk)
    cla_loss = F.cross_entropy(pred_cla, target_cla)

    #pred_spk = torch.log(pred_spk + 1e-7)
    #spk_loss = F.nll_loss(pred_spk, target_spk)
    
    #pred_cla = torch.log(pred_cla + 1e-7)
    #cla_loss = F.nll_loss(pred_cla, target_cla)
    total_loss = AWL(params, cla_loss, spk_loss) + score_loss
    return total_loss

def mse_ce_3t_noae(pred_score, target_score, pred_spk, target_spk, pred_cla, target_cla):

    score_loss = F.mse_loss(pred_score, target_score)
    spk_loss = F.cross_entropy(pred_spk, target_spk)
    cla_loss = F.cross_entropy(pred_cla, target_cla)

    #pred_spk = torch.log(pred_spk + 1e-7)
    #spk_loss = F.nll_loss(pred_spk, target_spk)
    
    #pred_cla = torch.log(pred_cla + 1e-7)
    #cla_loss = F.nll_loss(pred_cla, target_cla)
    
    total_loss = score_loss + 0.05 * cla_loss + spk_loss
    return total_loss, score_loss

def mse_ce_2t_noae(pred_score, target_score, pred_spk, target_spk, pred_cla, target_cla):

    score_loss = F.mse_loss(pred_score, target_score)
    spk_loss = F.cross_entropy(pred_spk, target_spk)
    total_loss = score_loss + spk_loss
    return total_loss, score_loss

def mse(recon, origin, pred_score, target_score):
    
    recon_loss = F.mse_loss(recon, origin)
    return recon_loss

def mse_ce(recon, origin, pred_score, target_score):
    
    recon_loss = F.mse_loss(recon, origin)
    score_loss = F.mse_loss(pred_score, target_score)
    total_loss = recon_loss + 0.2 * score_loss
    return total_loss

def mse_ce_2t_spk(recon, origin, pred_score, target_score, pred_spk, target_spk):
    
    recon_loss = F.mse_loss(recon, origin)
    score_loss = F.mse_loss(pred_score, target_score)
    spk_loss = F.cross_entropy(pred_spk, target_spk)
    total_loss = recon_loss + score_loss + 0.3 * spk_loss
    return total_loss

def mse_ce_3t(recon, origin, pred_score, target_score, pred_spk, target_spk, pred_cla, target_cla):
    
    recon_loss = F.mse_loss(recon, origin)
    score_loss = F.mse_loss(pred_score, target_score)
    spk_loss = F.cross_entropy(pred_spk, target_spk)
    cla_loss = F.cross_entropy(pred_cla, target_cla)
    #total_loss = recon_loss + 0.1 * (score_loss + 1.0 * cla_loss + spk_loss)
    total_loss = recon_loss + 0.6 * (score_loss + 0.8 * cla_loss + spk_loss)
    min_loss = score_loss
    #min_loss = recon_loss + 0.1 * (score_loss + 0.8 * cla_loss)
    #min_loss = recon_loss
    return total_loss, min_loss

def mse_ce_2t_sccla(recon, origin, pred_score, target_score, pred_spk, target_spk, pred_cla, target_cla):
    
    recon_loss = F.mse_loss(recon, origin)
    score_loss = F.mse_loss(pred_score, target_score)
    #spk_loss = F.cross_entropy(pred_spk, target_spk)
    cla_loss = F.cross_entropy(pred_cla, target_cla)
    total_loss = recon_loss + 0.1 * (score_loss + 1.0 * cla_loss)
    print(recon_loss, score_loss, cla_loss)
    #min_loss = score_loss
    min_loss = recon_loss
    #min_loss = recon_loss + 0.3 * (score_loss + 0.8 * cla_loss)
    return total_loss, min_loss

def mse_ce_2t_scspk(recon, origin, pred_score, target_score, pred_spk, target_spk, pred_cla, target_cla):
    
    recon_loss = F.mse_loss(recon, origin)
    score_loss = F.mse_loss(pred_score, target_score)
    spk_loss = F.cross_entropy(pred_spk, target_spk)
    #cla_loss = F.cross_entropy(pred_cla, target_cla)
    total_loss = recon_loss + 0.1 * (score_loss + spk_loss)
    print(recon_loss, score_loss, spk_loss)
    #min_loss = score_loss
    min_loss = recon_loss
    #min_loss = recon_loss + 0.3 * (score_loss + 0.8 * cla_loss)
    return total_loss, min_loss

def mse_ce_2t_spkcla(recon, origin, pred_score, target_score, pred_spk, target_spk, pred_cla, target_cla):
    
    recon_loss = F.mse_loss(recon, origin)
    #score_loss = F.mse_loss(pred_score, target_score)
    spk_loss = F.cross_entropy(pred_spk, target_spk)
    cla_loss = F.cross_entropy(pred_cla, target_cla)
    total_loss = recon_loss + 0.1 * (0.8 * cla_loss + spk_loss)
    min_loss = recon_loss
    #min_loss = recon_loss + 0.3 * (score_loss + 0.8 * cla_loss)
    return total_loss, min_loss


def mse_3t_mos(recon, origin, pred_score, target_score, pred_spk, target_spk, pred_cla, target_cla):
    
    recon_loss = F.mse_loss(recon, origin)
    score_loss = F.mse_loss(pred_score, target_score)
    total_loss = recon_loss + 0.6 * score_loss
    return total_loss, recon_loss

def mse_3t(recon, origin, pred_score, target_score, pred_spk, target_spk, pred_cla, target_cla):
    
    recon_loss = F.mse_loss(recon, origin)
    return recon_loss, recon_loss

def clip_bce(output_dict, target_dict):
    """Binary cross entropy loss.

    Args:
      output_dict: {'clipwise_output': (N, classes_num)}
      target_dict: {'target': (N, classes_num)}
    """
    return F.binary_cross_entropy(output_dict['clipwise_output'], target_dict['target'])

def clip_mse(output_dict, target_dict):
    
    return F.mse_loss(output_dict, target_dict)
    
def clip_mse_vcc(output_dict, frame, target_dict):
    """Binary cross entropy loss.

    Args:
      output_dict: {'clipwise_output': (N, classes_num)}
      target_dict: {'target': (N, classes_num)}
    """
    #return F.mse_loss(output_dict['clipwise_output'], target_dict['target'])
    #print("target", target_dict.size())
    #print("output", output_dict.size())

    return F.mse_loss(output_dict, target_dict)

def clip_frame_mse(clip, frame, target):
    
    clip_out = clip
    frame_out = frame  
    
    #clip_mse = torch.mean((clip_out - target) ** 2)
    clip_mse = (target - clip_out) ** 2
    #frame_mse = torch.mean((10 ** (target - 5)) * torch.mean((frame_out - target) ** 2, dim=1, keepdim=True))
    target = target.unsqueeze(1)
    #frame = frame_out - target
    frame_mse = 0.8 * torch.mean((target - frame_out) ** 2, dim=1)
    
    loss = clip_mse + frame_mse
    loss = torch.mean(loss)
    return loss

def clip_frame_mse_new(clip, frame, target):
    
    clip_out = clip
    frame_out = frame  
    
    #clip_mse = torch.mean((clip_out - target) ** 2)
    clip_mse = (target - clip_out) ** 2
   
    seq_len = frame_out.shape[1]
    
    #batch = frame_out.shape[0]
    #w = []
    #for i in range(batch):
    #    a = 10**(clip_out[i]-target[i])
    #    w.append(a)
    #w = torch.tensor(w).cuda()

    target = target.unsqueeze(1).repeat(1, seq_len)

    #frame_mse = w * torch.mean((target - frame_out) ** 2, dim=1)
    
    frame_mse = 1.0 * torch.mean((target - frame_out) ** 2, dim=1)
    
    loss = clip_mse + frame_mse
    loss = torch.mean(loss)
    return loss

def clipped_frame_mse_tau(clip, frame, target):
    
    clip_out = clip
    frame_out = frame 
    tau = 0.5

    #clip_mse = (target - clip_out) ** 2
    clip_mse = F.mse_loss(clip_out, target, reduction = 'none')
    clip_threshold = torch.abs(clip_out - target)>tau
    clip_mse = torch.mean(clip_threshold*clip_mse)


    seq_len = frame_out.shape[1]
    target = target.unsqueeze(1).repeat(1, seq_len)
    
    frame_mse = F.mse_loss(frame_out, target, reduction = 'none')
    frame_threshold = torch.abs(frame_out - target)>tau
    frame_mse = torch.mean(frame_threshold*frame_mse)
    
    loss = clip_mse + 0.9 * frame_mse
    #loss = torch.mean(loss)
    return loss

def clipped_frame_mse(clip, frame, target):
    
    clip_out = clip
    frame_out = frame 
    tau = 0.5

    #clip_mse = (target - clip_out) ** 2
    clip_mse = F.mse_loss(clip_out, target, reduction = 'none')
    clip_threshold = torch.abs(clip_out - target)>tau
    clip_mse = torch.mean(clip_threshold*clip_mse)


    seq_len = frame_out.shape[1]
    target = target.unsqueeze(1).repeat(1, seq_len)
    
    frame_mse = F.mse_loss(frame_out, target, reduction = 'none')
    frame_threshold = torch.abs(frame_out - target)>tau
    frame_mse = torch.mean(frame_threshold*frame_mse)
    #frame_mse = torch.mean(frame_mse)
    

    #frame = frame_out - target
    #frame_mse = 0.8 * torch.mean((target - frame_out) ** 2, dim=1)
    
    loss = clip_mse + 1.0 * frame_mse
    #loss = torch.mean(loss)
    return loss

#def frame_mse(y_true, y_pred):  # Customized loss function  (frame-level loss, the second term of equation 1 in the paper)
#    True_pesq=y_true[0,0]           
#        return (10**(True_pesq-4.5))*tf.reduce_mean((y_true-y_pred)**2)

def vae_loss_function(*args,
                  **kwargs):
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    :param args:
    :param kwargs:
    :return:
    """
    recons = args[0]
    #print(recons.shape)
    input = args[1]
    mu = args[2]
    log_var = args[3]

    kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
    recons_loss =F.mse_loss(recons, input)

    #kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1))

    loss = recons_loss + kld_weight * kld_loss
    #return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}
    #print("loss", loss)
    return loss

def vqvae_loss_function(*args,
                  **kwargs):
    """
    :param args:
    :param kwargs:
    :return:
    """
    recons = args[0]
    input = args[1]
    vq_loss = args[2]

    recons_loss = F.mse_loss(recons, input)

    loss = recons_loss + vq_loss
    #return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'VQ_Loss':vq_loss}
    return loss

def vqvae_3t_loss_function(*args,
                  **kwargs):
    """
    :param args:
    :param kwargs:
    :return:
    """
    recons = args[0]
    input = args[1]
    vq_loss = args[2]
    pred_score = args[4]
    pred_spk = args[5]
    pred_cla = args[6]
    target_score = args[7]
    target_spk = args[8]
    target_cla = args[9]


    recons_loss = F.mse_loss(recons, input)
    score_loss = F.mse_loss(pred_score, target_score)
    spk_loss = F.cross_entropy(pred_spk, target_spk)
    cla_loss = F.cross_entropy(pred_cla, target_cla)
    total_loss = recons_loss + vq_loss + 0.3 * (score_loss + 0.8 * cla_loss + spk_loss)
    min_loss = score_loss

    #loss = recons_loss + vq_loss
    return total_loss, min_loss

def vqvae_3t_ft_loss_function(*args,
                  **kwargs):
    """
    :param args:
    :param kwargs:
    :return:
    """
    recons = args[0]
    input = args[1]
    vq_loss = args[2]
    pred_score = args[4]
    pred_spk = args[5]
    pred_cla = args[6]
    target_score = args[7]
    target_spk = args[8]
    target_cla = args[9]


    recons_loss = F.mse_loss(recons, input)
    total_loss = recons_loss
    min_loss = recons_loss

    #loss = recons_loss + vq_loss
    return total_loss, min_loss

def mse_lcc(output_dict, frame, target_dict):

    mse = F.mse_loss(output_dict, target_dict)
    lcc = pearsonr(output_dict, target_dict)
    #print(lcc)
    total_loss = (1 - 0.09) * mse - 0.09 * lcc
    return total_loss


def get_loss_func(loss_type):
    if loss_type == 'clip_bce':
        return clip_bce
    elif loss_type == 'clip_mse':
        return clip_mse
    elif loss_type == 'clip_mse_vcc':
        return clip_mse_vcc
    elif loss_type == 'clip_frame_mse':
        return clip_frame_mse
    elif loss_type == 'clip_frame_mse_new':
        return clip_frame_mse_new
    elif loss_type == 'clipped_frame_mse':
        return clipped_frame_mse
    elif loss_type == 'clipped_frame_mse_tau':
        return clipped_frame_mse_tau
    elif loss_type == 'vae_loss_function':
        return vae_loss_function
    elif loss_type == 'vqvae_loss_function':
        return vqvae_loss_function
    elif loss_type == 'vqvae_3t_loss_function':
        return vqvae_3t_loss_function
    elif loss_type == 'vqvae_3t_ft_loss_function':
        return vqvae_3t_ft_loss_function
    elif loss_type == 'mse_ce':
        return mse_ce
    elif loss_type == 'mse_ce_2t_spk':
        return mse_ce_2t_spk
    elif loss_type == 'mse_ce_3t':
        return mse_ce_3t
    elif loss_type == 'mse_ce_3t_noae':
        return mse_ce_3t_noae
    elif loss_type == 'mse_ce_3t_noae_awl':
        return mse_ce_3t_noae_awl
    elif loss_type == 'mse_ce_2t_noae':
        return mse_ce_2t_noae
    elif loss_type == 'mse':
        return mse
    elif loss_type == 'mse_lcc':
        return mse_lcc
    elif loss_type == 'mse_3t':
        return mse_3t
    elif loss_type == 'mse_3t_mos':
        return mse_3t_mos
    elif loss_type == 'mse_ce_2t_sccla':
        return mse_ce_2t_sccla
    elif loss_type == 'mse_ce_2t_scspk':
        return mse_ce_2t_scspk
    elif loss_type == 'mse_ce_2t_spkcla':
        return mse_ce_2t_spkcla



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
    



