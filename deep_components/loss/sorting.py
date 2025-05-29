import torch
import torch.nn.functional as F

def bl_matmul(A, B):
    return torch.einsum('mij,jk->mik', A, B)

# neuralsort(s): M x n x n
def neuralsort(s, tau=1):
    s = s.unsqueeze(-1)
    A_s = torch.abs(s - s.transpose(1, 2))
    # As_ij = |s_i - s_j|

    n = s.size(1)
    one = torch.ones((n, 1), dtype=torch.float32, device=s.device)

    B = bl_matmul(A_s, one @ one.T)
    # B_:k = (A_s)(one)

    K = torch.arange(n, device=s.device) + 1
    # K_k = k

    C = bl_matmul(s, (n + 1 - 2 * K).float().unsqueeze(0))
    # C_:k = (n + 1 - 2k)s

    P = (C - B).transpose(1, 2)
    # P_k: = (n + 1 - 2k)s - (A_s)(one)

    P = F.softmax(P / tau, dim=-1)
    # P_k: = softmax( ((n + 1 - 2k)s - (A_s)(one)) / tau )

    return P


def soft_sort(s, tau):
    s = s.unsqueeze(-1)
    s_sorted, _ = torch.sort(s, descending=True, dim=1)
    pairwise_distances = -torch.abs(s.transpose(1, 2) - s_sorted)
    P_hat = F.softmax(pairwise_distances / tau, dim=-1)
    return P_hat

class NeuralSortNet:
    def __init__(self, tau, descending=True):
        self.tau = tau
        self.descending = descending
        self.net = neuralsort

    def forward(self, x):
        if self.descending:
            return self.net(x, self.tau)
        else:
            return self.net(x, self.tau)[:, :, ::-1]

    @staticmethod
    def get_default_config():
        config = {}
        config['tau'] = 1
        return config

class SoftSortNet:
    def __init__(self, tau, descending=True):
        self.tau = tau
        self.descending = descending
        self.net = soft_sort

    def forward(self, x):
        if self.descending:
            return self.net(x, self.tau)
        else:
            return self.net(x, self.tau)[:, :, ::-1]

    @staticmethod
    def get_default_config():
        config = {}
        config['tau'] = 1
        return config

class SortNet:
    def __init__(self, sort_op, reverse=False, config=None):
        self.sort_op = sort_op
        self.config = config
        self.reverse = reverse
        if sort_op == 'neural_sort':
            self.net = NeuralSortNet(tau=self.config['tau'], descending=True)
        elif sort_op == 'soft_sort':
            self.net = SoftSortNet(tau=self.config['tau'], descending=True)
        else:
            raise NotImplementedError(f'[ERROR] sort_op `{sort_op}` unknown')

    def forward(self, x):
        permutation_matrix = self.net.forward(x)
        if (self.reverse and self.sort_op in ['neural_sort', 'soft_sort']):
            return torch.flip(permutation_matrix, dims=[-2])
        else:
            return permutation_matrix

    @staticmethod
    def get_default_config(sort_op):
        if sort_op == 'neural_sort':
            return NeuralSortNet.get_default_config()
        elif sort_op == 'soft_sort':
            return SoftSortNet.get_default_config()