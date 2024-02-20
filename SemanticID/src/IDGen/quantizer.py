import torch
from torch import nn
from torch.nn import functional as F

# import distributed as dist_fn
from torch import distributed as dist


from IPython import embed


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def all_reduce(tensor, op=dist.ReduceOp.SUM):
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=op)
    return tensor


class Quantize(nn.Module):
    '''
    This is the EMA updating Quantization Module.
    Since it is using EMA, it need to select code by using L1 distance calculation.
    Using dot production may cause error here, since the code selection and update are not aligned.
    '''
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        # embed = torch.randn(dim, n_embed)
        embed = torch.FloatTensor(dim, n_embed)
        nn.init.xavier_normal_(embed)

        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.ones(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)

        # L2 distance selection
        # print('You are using L2 distance to select ids. Take care!!!')
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)

        # dot production distance selection
        # print('You are using dot production to select ids. Take care!!!')
        # dot_product = flatten @ self.embed
        # _, embed_ind = dot_product.max(1)

        # print(embed_ind)

        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            all_reduce(embed_onehot_sum)
            all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()

            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class GBQuantize(nn.Module):
    '''
    This is the Gumble Softmax updating Quantization Module.
    '''
    def __init__(self, dim, n_embed, temperature=1):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.temperature = temperature

        self.embed = nn.Parameter(torch.FloatTensor(n_embed, dim))
        nn.init.xavier_normal_(self.embed)

    def forward(self, input):

        flatten = input.reshape(-1, self.dim)

        dot_product = flatten @ self.embed.t()
        softmax_dot_product = F.softmax(dot_product / self.temperature, dim=1)

        # hard quantize
        _, embed_ind = dot_product.max(1)
        hard_embedding = self.embed[embed_ind]

        # soft quantize
        soft_embedding = torch.mm(softmax_dot_product, self.embed)

        quantize = soft_embedding + (hard_embedding - soft_embedding).detach()

        return quantize, 0, embed_ind
