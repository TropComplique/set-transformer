import torch

    
def get_random_datasets(batch_size, k, min_size, max_size):
    """
    Generates a random mixture of k gaussians in the plane.
    Means are in range [-4, 4]x[-4, 4].
    Standard deviation is 0.3.

    Arguments: 
        batch_size: an integer, number of datasets. 
        k: an integer, number of clusters in each dataset.
        min_size, max_size: integers.
    Returns:
        data: a float tensor with shape [batch_size, n, 2],
            where min_size <= n <= max_size.
        params: a dict with the following keys
            'means': a float tensor with shape [batch_size, k, 2],
            'pi': a float tensor with shape [batch_size, k].
    """

    # dataset size
    n = torch.randint(min_size, max_size + 1, size=[], dtype=torch.int32)
    # shape []

    # parameters of gaussians
    centers = 4.0*(2.0*torch.rand(batch_size, k, 2) - 1.0)
    sigmas = torch.Tensor([0.3]).unsqueeze(0).unsqueeze(0).repeat([batch_size, k, 2])
    # they have shape [batch_size, k, 2]

    # probabilities to be in different clusters
    pi = torch.distributions.Dirichlet(torch.Tensor(k*[1.0])).sample((batch_size,))
    # shape [batch_size, k]

    # assignments to each cluster
    labels = torch.distributions.categorical.Categorical(probs=pi).sample((n,)).t()
    # shape [batch_size, n]

    labels = labels.unsqueeze(2).repeat([1, 1, 2])
    selected_centers = torch.gather(centers, 1, labels)
    selected_sigmas = torch.gather(sigmas, 1, labels)

    data = torch.normal(selected_centers, selected_sigmas)
    params = {'means': centers, 'pi': pi}
    return data, params
