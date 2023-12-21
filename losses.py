import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import traceback
import datetime
import torch
import time


class GroupContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
        self.register_buffer("positives_mask", (~torch.eye(batch_size * 1, batch_size * 1, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """

        self.negatives_mask[:len(emb_i), :len(emb_j)] = False
        self.negatives_mask[len(emb_i):, len(emb_j):] = False

        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2).cuda()

        pos_similarity_matrix = similarity_matrix[:len(emb_i), :len(emb_j)].cuda()
        neg_similarity_matrix = similarity_matrix[len(emb_i):, len(emb_j):].cuda()

        pos_similarity_matrix = pos_similarity_matrix * self.positives_mask
        sim_ij=torch.sum(pos_similarity_matrix,dim=1)/(len(neg_similarity_matrix)-1)

        neg_similarity_matrix = neg_similarity_matrix * self.positives_mask
        sim_ji = torch.sum(neg_similarity_matrix, dim=1)/(len(neg_similarity_matrix)-1)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        numerator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(numerator / (numerator + torch.sum(denominator, dim=1)))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)

        return loss


def weighted_contrastive_loss(features_images, features_augmentations, tau, annotator_matrices, mode_in=True):
    """

    Weighted contrastive loss (one sided). If mode_in is set to True, Lin is invoked as the loss, otherwise Lout.
    These losses correspond to the expressions as per supervised contrastive learning
    in https://proceedings.neurips.cc/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf.

    Both features of shape (B, D, C), where C is the feature length, D is the number of distortions. A batch of
    annotator_matrices of shape (B, D, D).

    """

    # Normalizing all features with l2 norm
    eps = 1e-8 # for computational stability
    norm_images = torch.linalg.norm(features_images, dim=-1)
    norm_augmentations = torch.linalg.norm(features_augmentations, dim=-1)
    norm_images = torch.max(norm_images, eps * torch.ones_like(norm_images))
    norm_augmentations = torch.max(norm_augmentations, eps * torch.ones_like(norm_augmentations))
    normalized_features_images = features_images/norm_images.unsqueeze(dim=-1)
    normalized_features_augmentations = features_augmentations/norm_augmentations.unsqueeze(dim=-1)

    # Computing loss for pairs
    feat_distances = torch.bmm(normalized_features_images, torch.transpose(normalized_features_augmentations, dim0=1, dim1=2)) / tau  # (B, 1)
    alpha = 2.0 - 2.0 / (1 + annotator_matrices ** 2)
    term_pos = alpha * torch.exp(feat_distances)
    term_neg = torch.exp(feat_distances)

    # Choosing Lin or Lout as the training loss
    if mode_in:
        loss1 = torch.divide(term_pos.sum(-1), term_neg.sum(-1))
        loss1 = -torch.log(loss1)
        loss1 = loss1.mean()
        loss2 = torch.divide(term_pos.sum(-2), term_neg.sum(-2))
        loss2 = -torch.log(loss2)
        loss2 = loss2.mean()
        loss = loss1 + loss2

    else:
        loss1 = - alpha * (torch.log(term_neg) - torch.log(term_neg.sum(-1))[:,:,None])
        loss2 = - alpha * (torch.log(term_neg) - torch.log(term_neg.sum(-2))[:,None])
        loss = loss1.mean() + loss2.mean()

    return loss


# Testing the quality aware contrastive loss
def test_qacl():
    ssim = torch.tril(torch.rand(5, 9, 9), diagonal=-1)
    ssim = ssim + torch.transpose(ssim, dim0=1, dim1=2)  # To get full matrix from lower triangular matrix
    ssim = torch.exp(-ssim)

    feat1 = torch.rand(5, 9, 128)
    feat2 = torch.rand(5, 9, 128)

    losses = weighted_contrastive_loss(feat1, feat2, 0.2, ssim)
    print(losses)
    return


# Testing the group contrastive loss
def test_gcl():
    pseudo_labels=torch.rand(16,1)
    f_feat=torch.rand(16,256)
    batch_size = 16

    idx = np.argsort(pseudo_labels.cpu(), axis=0)
    f_pos_feat = []
    f_neg_feat = []

    for n in range( batch_size // 4):
        try:
            f_pos_feat.append(f_feat[idx[n]])
            f_neg_feat.append(f_feat[idx[-n - 1]])
        except:
            continue

    f_pos_feat = torch.squeeze(torch.stack(f_pos_feat), dim=1)
    f_neg_feat = torch.squeeze(torch.stack(f_neg_feat), dim=1)

    loss_fn = GroupContrastiveLoss(f_pos_feat.shape[0], 1).cuda()
    loss = loss_fn(f_neg_feat, f_pos_feat)
    print(loss)

    return


def main():
    test_qacl()
    # test_gcl()
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
