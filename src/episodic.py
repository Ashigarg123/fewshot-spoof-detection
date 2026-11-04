import numpy as np
import torch
from torch.nn import functional as F


class EpisodicBatchSampler:
    def __init__(self, labels, classes_per_it, num_samples, iterations,
                 always_include_class_zero=False):
        self.classes_per_it = classes_per_it
        self.samples_per_class = num_samples
        self.iterations = iterations
        self.always_include_class_zero = always_include_class_zero

        classes, counts = np.unique(labels, return_counts=True)
        self.classes = torch.as_tensor(classes, dtype=torch.long)
        self.indexes = torch.full((len(classes), max(counts)), float('nan'))
        self.counts = torch.zeros(len(classes), dtype=torch.long)
        for sample_index, label in enumerate(labels):
            class_index = np.argwhere(classes == label).item()
            offset = self.counts[class_index].item()
            self.indexes[class_index, offset] = sample_index
            self.counts[class_index] += 1

    def __iter__(self):
        for _ in range(self.iterations):
            if self.always_include_class_zero:
                zero_index = (self.classes == 0).nonzero(as_tuple=True)[0].item()
                other_indices = (self.classes != 0).nonzero(as_tuple=True)[0]
                selected = other_indices[
                    torch.randperm(len(other_indices))[:self.classes_per_it - 1]
                ]
                class_indices = torch.cat((torch.tensor([zero_index]), selected))
            else:
                class_indices = torch.randperm(len(self.classes))[:self.classes_per_it]

            batch = torch.empty(
                self.samples_per_class * self.classes_per_it,
                dtype=torch.long,
            )
            for position, class_index in enumerate(class_indices):
                output_slice = slice(
                    position * self.samples_per_class,
                    (position + 1) * self.samples_per_class,
                )
                selected = torch.randperm(self.counts[class_index])[:self.samples_per_class]
                batch[output_slice] = self.indexes[class_index, selected].long()
            yield batch[torch.randperm(len(batch))]

    def __len__(self):
        return self.iterations


def euclidean_dist(queries, prototypes):
    if queries.size(1) != prototypes.size(1):
        raise ValueError('query and prototype dimensions do not match')
    return torch.pow(queries[:, None, :] - prototypes[None, :, :], 2).sum(2)


def prototypical_loss(embeddings, targets, n_support, prototype_aggregator=None):
    targets_cpu = targets.cpu()
    classes = torch.unique(targets_cpu)
    n_classes = len(classes)
    n_query = targets_cpu.eq(classes[0].item()).sum().item() - n_support

    support_indices = [
        targets_cpu.eq(label).nonzero()[:n_support].squeeze(1)
        for label in classes
    ]
    if prototype_aggregator is None:
        embeddings_cpu = embeddings.cpu()
        prototypes = torch.stack([
            embeddings_cpu[indices].mean(0) for indices in support_indices
        ])
    else:
        prototypes = torch.stack([
            prototype_aggregator(embeddings[indices].unsqueeze(1)).squeeze(0)
            for indices in support_indices
        ]).cpu()

    query_indices = torch.stack([
        targets_cpu.eq(label).nonzero()[n_support:] for label in classes
    ]).view(-1)
    queries = embeddings.cpu()[query_indices]
    distances = euclidean_dist(queries, prototypes)
    log_probabilities = F.log_softmax(-distances, dim=1).view(n_classes, n_query, -1)
    probabilities = F.softmax(-distances, dim=1).view(n_classes, n_query, -1)
    expected = torch.arange(n_classes).view(n_classes, 1, 1)
    expected = expected.expand(n_classes, n_query, 1).long()

    loss = -log_probabilities.gather(2, expected).squeeze().view(-1).mean()
    predictions = log_probabilities.max(2)[1]
    accuracy = predictions.eq(expected.squeeze(2)).float().mean()
    return loss, accuracy, query_indices, probabilities
