from pytorch_lightning.core import LightningModule, LightningDataModule
import torch.nn as nn
import torch
from transformers import RobertaTokenizer, RobertaModel
from data.datasets import (
    WikipediaTextDatasetParagraphsSentences,
    WikipediaTextDatasetParagraphsSentencesTest
)
from torch.utils.data.dataloader import DataLoader
import math
from datetime import datetime
import os
from models.SDR_utils import MPerClassSamplerDeter
from pytorch_metric_learning.samplers import MPerClassSampler
from data.data_utils import reco_sentence_collate, reco_sentence_test_collate
from functools import partial
from pytorch_metric_learning import miners, losses, reducers
from pytorch_metric_learning.distances import CosineSimilarity
import pickle as pkl
import faiss
from models.eval_metrics import *

class RTN(LightningModule):
    def __init__(self, hparams):
        super(RTN, self).__init__()
        self.hparams.update(vars(hparams))
        path = hparams.default_root_dir + hparams.dataset_name
        dt_string = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
        path = os.path.join(path, dt_string)
        os.makedirs(path, exist_ok=True)
        self.hparams.hparams_dir = path
        self.d_model = 768
        self.hparams.max_input_len = 512
        self.model = RobertaModel.from_pretrained('roberta-base')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        #self.transformer = nn.Transformer(d_model=self.d_model, nhead=8, num_encoder_layers=6,
        #                               num_decoder_layers=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pos_emb = PositionalEncoding(self.d_model, 512)
        #self.classifier = nn.Linear(2*self.d_model, 1)
        self.metric = CosineSimilarity()
        pos_margin, neg_margin = 1, 0
        self.loss_func = losses.ContrastiveLoss(pos_margin=pos_margin, neg_margin=neg_margin, distance=self.metric)
        self.miner_func = miners.MultiSimilarityMiner()
        self.tracks = {}

    def forward(self, batch):
        paragraphs = []
        sample_labels = batch[6]
        #for i in range(len(batch[0])):
        #    #section = self.tokenizer(section, return_tensors="pt")
        #    # extract one vector representation for each paragraph
        #    out = self.model(input_ids=batch[0][i], attention_mask=batch[1][i])
        #    paragraphs.append(out)
        out = self.model(input_ids=batch[0], attention_mask=batch[1])
        total_num_sections = 0
        sections_per_article = []
        paragraph_mask = []
        for i in range(len(batch[3])):
            num_secs = len(batch[3][i])
            paragraph_mask.append(torch.zeros(num_secs, device=self.device))
            this_articles_sections = out['last_hidden_state'][total_num_sections:total_num_sections+num_secs]
            sections_per_article.append(this_articles_sections)
            total_num_sections += num_secs
        out = torch.nn.utils.rnn.pad_sequence(sections_per_article, padding_value=self.tokenizer.pad_token_id)
        out = out.mean(dim=2)
        paragraph_mask = torch.nn.utils.rnn.pad_sequence(paragraph_mask, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        out = self.pos_emb(out) + out
        out = self.transformer(out, src_key_padding_mask=paragraph_mask)
        meaned_sentences = out.mean(0)
        miner_output = list(self.miner_func(meaned_sentences, sample_labels))
        loss = self.loss_func(meaned_sentences, sample_labels, miner_output)
        #out = self.classifier(out)
        return loss, meaned_sentences, batch[2]


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        loss, _, _ = self(train_batch)
        return loss

    def validation_step(self, val_batch, batch_idx):
        val_loss, _, _ = self(val_batch)
        self.tracks['loss'] = val_loss.detach()
        return val_loss


    def test_step(self, test_batch, batch_idx):
        test_loss, outputs, titles = self(test_batch)
        return outputs, titles

    def test_epoch_end(self, outputs):
        device = 'cpu'
        if not os.path.isfile("test_embeds_dev_run.pkl"):
            pkl.dump(outputs, open("test_embeds_dev_run.pkl", 'wb'))
        else:
            outputs = pkl.load(open("test_embeds_dev_run.pkl", "rb"))
        res = faiss.StandardGpuResources()
        gt_path = "./data/datasets/{}/gt".format(self.hparams.dataset_name)
        gt = pkl.load(open(gt_path, "rb"))
        doc_embeddings = []
        doc_titles = []
        article2idx = {}
        article2embedding = {}
        flattened_outputs = []
        for i, tup in enumerate(outputs):
            doc_embeddings.append(tup[0].to(device))
            doc_titles.extend(tup[1])
            flattened_outputs.extend(list(zip(tup[0], tup[1])))
            for j, title in enumerate(tup[1]):
                article2idx[title] = i*len(tup[1])+j
                #article2embedding[title] = tup[0][j]

        outputs = flattened_outputs
        doc_embeddings = torch.cat(doc_embeddings, dim=0)
        index = faiss.IndexFlatL2(len(doc_embeddings[0]))  # build the index
        #index = faiss.index_cpu_to_gpu(res, 0, index)
        print(index.is_trained)
        index.add(doc_embeddings)  # add vectors to the index
        print(index.ntotal)
        gt_article_embeds = []
        not_found_counter = 0
        gt_as_ids = {}
        for article in gt:
            lookup = article.replace("&",
                                      "&amp;") if "amp;" not in article and article not in doc_titles else article
            if lookup in doc_titles:
                idx = doc_titles.index(lookup)
                gt_article_embeds.append(doc_embeddings[idx])
                #gt_as_ids[article] = torch.tensor([doc_titles.index(sim_art) for sim_art in gt[article]])
            else:
                print(lookup)
                not_found_counter += 1
        print('Did not find {} of {} articles in gt.'.format(not_found_counter, len(gt)))
        gt_article_embeds = torch.stack(gt_article_embeds, dim=0)
        D, I = index.search(gt_article_embeds, len(doc_embeddings))
        recos = []
        for i, article in enumerate(gt):
            lookup = article.replace("&",
                                     "&amp;") if "amp;" not in article and article not in doc_titles else article
            if lookup in doc_titles:
                recos.append((article2idx[lookup], I[i]))
            else:
                continue
        output_path = 'test'
        evaluate_wiki_recos(recos, output_path, gt_path, outputs)




class SDRDataset(LightningDataModule):
    def __init__(self, hparams, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.hparams.update(vars(hparams))

    def train_dataloader(self):
        sampler = MPerClassSampler(
            self.train_dataset.labels,
            self.hparams.train_batch_size/2,
            batch_size=self.hparams.train_batch_size,
            length_before_new_iter=self.hparams.limit_train_batches,
        )

        loader = DataLoader(
            self.train_dataset,
            num_workers=self.hparams.num_data_workers,
            sampler=sampler,
            batch_size=self.hparams.train_batch_size,
            collate_fn=partial(reco_sentence_collate, tokenizer=self.tokenizer,),
        )
        return loader

    def val_dataloader(self):
        sampler = MPerClassSamplerDeter(
            self.val_dataset.labels,
            self.hparams.val_batch_size/2,
            length_before_new_iter=self.hparams.limit_val_indices_batches,
            batch_size=self.hparams.val_batch_size,
        )

        loader = DataLoader(
            self.val_dataset,
            num_workers=self.hparams.num_data_workers,
            sampler=sampler,
            batch_size=self.hparams.val_batch_size,
            collate_fn=partial(reco_sentence_collate, tokenizer=self.tokenizer,),
            shuffle=False,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            num_workers=self.hparams.num_data_workers,
            batch_size=self.hparams.test_batch_size,
            collate_fn=partial(reco_sentence_test_collate, tokenizer=self.tokenizer,),
            shuffle=False,
        )
        return loader

    def prepare_data(self):
        return

    def setup(self, stage):
        self.dataset_name = self.hparams.dataset_name
        block_size = (
            self.hparams.block_size
            if hasattr(self.hparams, "block_size")
            and self.hparams.block_size > 0
            and self.hparams.block_size < self.tokenizer.max_len_single_sentence
            else self.tokenizer.max_len_single_sentence
        )
        self.train_dataset = WikipediaTextDatasetParagraphsSentences(
            tokenizer=self.tokenizer,
            hparams=self.hparams,
            dataset_name=self.dataset_name,
            block_size=block_size,
            mode="train",
        )
        self.val_dataset = WikipediaTextDatasetParagraphsSentences(
            tokenizer=self.tokenizer,
            hparams=self.hparams,
            dataset_name=self.dataset_name,
            block_size=block_size,
            mode="val",
        )
        #self.val_dataset.indices_map = self.val_dataset.indices_map[: self.hparams.limit_val_indices_batches]
        #self.val_dataset.labels = self.val_dataset.labels[: self.hparams.limit_val_indices_batches]

        self.test_dataset = WikipediaTextDatasetParagraphsSentencesTest(
            tokenizer=self.tokenizer,
            hparams=self.hparams,
            dataset_name=self.dataset_name,
            block_size=block_size,
            mode="test",
        )
        print('data prepared')






class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class ContrastiveTransformations(object):

    def __init__(self, base_transforms='paragraph_insert', n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]

    def _shuffle_paragraphs(self):
        print('here')
