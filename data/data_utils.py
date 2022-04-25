import pickle
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence



def get_gt_seeds_titles(titles=None, dataset_name="wines"):
    idxs = None
    gt_path = f"data/datasets/{dataset_name}/gt"
    popular_titles = list(pickle.load(open(gt_path, "rb")).keys())
    if titles != None:
        idxs = [titles.index(pop_title) for pop_title in popular_titles if pop_title in titles]
    return popular_titles, idxs, gt_path


def reco_sentence_test_collate(examples: List[torch.Tensor], tokenizer):
    sections_ids = []
    sections_masks = []
    for i in examples:
        sections_ids.extend(i[0])
        sections_masks.extend(i[1])
    return (
        pad_sequence(sections_ids, padding_value=tokenizer.pad_token_id),
        pad_sequence(sections_masks, padding_value=tokenizer.pad_token_id),
        [i[2] for i in examples],
        [i[3] for i in examples],
        [i[4] for i in examples],
        [i[5] for i in examples],
        torch.tensor([i[6] for i in examples]),
        [i[7] for i in examples]
    )


def reco_sentence_collate(examples: List[torch.Tensor], tokenizer):
    sections_ids = []
    sections_masks = []
    for i in examples:
        sections_ids.extend(i[0])
        sections_masks.extend(i[1])
    return (
        pad_sequence(sections_ids, padding_value=tokenizer.pad_token_id),
        pad_sequence(sections_masks, padding_value=tokenizer.pad_token_id),
        [i[2] for i in examples],
        [i[3] for i in examples],
        [i[4] for i in examples],
        [i[5] for i in examples],
        torch.tensor([i[6] for i in examples])
    )

def coherence_sentence_collate(examples: List[torch.Tensor], tokenizer):
    first_sent = [
        pad_sequence([i[0][0] for i in examples], batch_first=True, padding_value=tokenizer.pad_token_id),
        [i[0][2] for i in examples],
        [i[0][3] for i in examples],
        [i[0][4] for i in examples],
        [i[0][5] for i in examples],
        [i[0][6] for i in examples],
        [i[0][7] for i in examples],
        torch.tensor([i[0][8] for i in examples]),
    ]
    for n in range(1, len(examples[0])):
        first_sent[0] = torch.cat((first_sent[0], pad_sequence([i[n][0] for i in examples], batch_first=True, padding_value=tokenizer.pad_token_id)))
        first_sent[1].extend([i[n][2] for i in examples])
        first_sent[2].extend([i[n][3] for i in examples])
        first_sent[3].extend([i[n][4] for i in examples])
        first_sent[4].extend([i[n][5] for i in examples])
        first_sent[5].extend([i[n][6] for i in examples])
        first_sent[6].extend([i[n][7] for i in examples])
        first_sent[7] = torch.cat((first_sent[7], torch.tensor([i[n][8] for i in examples])))

    return tuple(first_sent)

def coherence_sentence_test_collate(examples: List[torch.Tensor], tokenizer):
    examples_ = []
    for example in examples:
        sections = []
        for section in example:
            if section == []:
                continue
            first_sent = [
                pad_sequence([i[0][0] for i in examples], batch_first=True, padding_value=tokenizer.pad_token_id),
                [i[0][2] for i in examples],
                [i[0][3] for i in examples],
                [i[0][4] for i in examples],
                [i[0][5] for i in examples],
                [i[0][6] for i in examples],
                [i[0][7] for i in examples],
                torch.tensor([i[0][8] for i in examples]),
            ]
            for n in range(1, len(examples[0])):
                first_sent[0] = torch.cat((first_sent[0], pad_sequence([i[n][0] for i in examples], batch_first=True,
                                                                       padding_value=tokenizer.pad_token_id)))
                first_sent[1].extend([i[n][2] for i in examples])
                first_sent[2].extend([i[n][3] for i in examples])
                first_sent[3].extend([i[n][4] for i in examples])
                first_sent[4].extend([i[n][5] for i in examples])
                first_sent[5].extend([i[n][6] for i in examples])
                first_sent[6].extend([i[n][7] for i in examples])
                first_sent[7] = torch.cat((first_sent[7], torch.tensor([i[n][8] for i in examples])))
            sections.append(tuple(first_sent))
        examples_.append(sections)
    return examples_

def NPP_sentence_collate(examples: List[torch.Tensor]):

    padded_seq = pad_sequence([i[0] for i in examples], padding_value=0)
    true_sentence_vectors = pad_sequence([i[1] for i in examples], padding_value=0)
    seq_mask = pad_sequence([i[2] for i in examples], padding_value=0, batch_first=True)
    padded_token_types = pad_sequence([i[3] for i in examples], padding_value=0)
    collated_examples = (
        padded_seq,
        true_sentence_vectors,
        seq_mask,
        padded_token_types,
        [i[8] for i in examples],
        torch.tensor([i[7] for i in examples]),
    )
    return collated_examples


def NPP_sentence_collate_test(examples: List[torch.Tensor]):

    padded_seq = pad_sequence([i[0] for i in examples], padding_value=0)
    #true_sentence_vectors = pad_sequence([i[1] for i in examples], padding_value=0)
    seq_mask = pad_sequence([i[1] for i in examples], padding_value=0, batch_first=True)
    padded_token_types = pad_sequence([i[2] for i in examples], padding_value=0)
    collated_examples = (
        padded_seq,
        padded_seq,
        seq_mask,
        padded_token_types,
        [i[7] for i in examples],
        torch.tensor(
            [i[6] for i in examples]),
        [i[3] for i in examples],
    )
    return collated_examples


def raw_data_link(dataset_name):
    if dataset_name == "wines":
        return "https://zenodo.org/record/4812960/files/wines.txt?download=1"
    if dataset_name == "video_games":
        return "https://zenodo.org/record/4812962/files/video_games.txt?download=1"
