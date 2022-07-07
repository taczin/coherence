from pytorch_lightning.core import LightningModule, LightningDataModule
import torch.nn as nn
import torch
from transformers import RobertaTokenizer, RobertaModel
from data.datasets import (
    WikipediaTextDatasetParagraphsSentences,
    WikipediaTextDatasetParagraphsSentencesTest,
    WikipediaTextDatasetParagraphOrder,
    XLSumDatasetParagraphOrder,
    XLSumDatasetParagraphOrderTest
)
from torch.utils.data.dataloader import DataLoader
import math
from datetime import datetime
import os
from models.SDR_utils import MPerClassSamplerDeter
from pytorch_metric_learning.samplers import MPerClassSampler
from data.data_utils import reco_sentence_collate, reco_sentence_test_collate, NPP_sentence_collate, NPP_sentence_collate_test
from functools import partial
from pytorch_metric_learning import miners, losses, reducers
from pytorch_metric_learning.distances import CosineSimilarity
import pickle as pkl
import faiss
from models.eval_metrics import *
from utils import eval_metrics
from sentence_transformers import SentenceTransformer

class NextParagraphPrediction(LightningModule):
    def __init__(self, hparams):
        super(NextParagraphPrediction, self).__init__()
        self.hparams.update(vars(hparams))
        path = hparams.default_root_dir + hparams.dataset_name
        dt_string = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
        path = os.path.join(path, dt_string)
        os.makedirs(path, exist_ok=True)
        self.hparams.hparams_dir = path
        self.d_model = 384
        self.hparams.max_input_len = 512
        self.d_hid = 2048
        self.dropout = 0.1
        self.nhead = 8
        self.num_layers = 2
        #self.model = RobertaModel.from_pretrained('roberta-base')
        # sentence embedding model
        #self.model = SentenceTransformer('all-MiniLM-L6-v2')
        #self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.tokenizer = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.tokenizer.max_len_sentences_pair = 512
        #self.transformer = nn.Transformer(d_model=self.d_model, nhead=8, num_encoder_layers=6,
        #                               num_decoder_layers=0)
        # need positional embeddings for sentences
        self.pos_emb = PositionalEncoding(self.d_model, self.hparams.max_input_len)
        # need equivalent of token type embeddings on sentence level
        self.sentence_type_emb = nn.Embedding(2, self.d_model)
        # need to normalize the summed embeddings before feeding to model
        self.norm = nn.LayerNorm(self.d_model)
        # need entire BERT-like transformer model
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.d_hid,
                                                   dropout=self.dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        # need MLM head for masked sentence prediction + loss function (cosine loss or metric learning loss)
        self.mlm_loss_func = nn.MSELoss()
        # need paragraph order head + loss function (binary cross-entropy loss)
        self.po_cls = nn.Linear(self.d_model, 1)
        self.po_loss_func = nn.BCEWithLogitsLoss()


        #self.classifier = nn.Linear(2*self.d_model, 1)
        #self.metric = CosineSimilarity()
        #pos_margin, neg_margin = 1, 0
        #self.loss_func = losses.ContrastiveLoss(pos_margin=pos_margin, neg_margin=neg_margin, distance=self.metric)
        #self.loss_func = torch.nn.BCEWithLogitsLoss
        #self.miner_func = miners.MultiSimilarityMiner()
        self.tracks = {}

    def forward(self, batch):
        paragraphs = []
        sample_labels = batch[4]
        #for i in range(len(batch[0])):
        #    #section = self.tokenizer(section, return_tensors="pt")
        #    # extract one vector representation for each paragraph
        #    out = self.model(input_ids=batch[0][i], attention_mask=batch[1][i])
        #    paragraphs.append(out)
        #pos_embeddings = self.pos_emb(torch.arange(batch[3].shape[1]).repeat((batch[3].shape[0], 1)))
        # adds positional embeddings to the input embeddings
        embeddings = self.pos_emb(batch[0])
        sent_type_embeddings = self.sentence_type_emb(batch[2].int())
        # the input embeddings are made up of the output of the sentence embedder, the sentence type embeddings and the
        # positional embeddings
        embeddings = embeddings+sent_type_embeddings
        embeddings = self.norm(embeddings)
        #out = self.model(input_ids=batch[0], attention_mask=batch[1])

        out = self.transformer(embeddings, src_key_padding_mask=batch[2])
        # retrieve masked vector
        mlm_loss = self.mlm_loss_func(out, batch[1])
        # alternatively take cls vector output instead of mean
        #meaned_sentences = out.mean(0)
        meaned_sentences = out[0]
        po_loss = self.po_loss_func(self.po_cls(meaned_sentences).squeeze(-1), batch[5].float())
        total_loss = mlm_loss + po_loss
        return total_loss, (mlm_loss, po_loss), meaned_sentences


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        loss, (mlm_loss, po_loss), _ = self(train_batch)
        self.log("training",
                 {"total_loss": loss.detach(), "mlm_loss": mlm_loss.detach(), "po_loss": po_loss.detach()}, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, (mlm_loss, po_loss), _ = self(val_batch)
        self.log("validation",
                 {"total_loss": loss.detach(), "mlm_loss": mlm_loss.detach(), "po_loss": po_loss.detach()})
        return loss


    def test_step(self, test_batch, batch_idx):
        loss, (mlm_loss, po_loss), outputs = self(test_batch)
        self.log("validation",
                 {"total_loss": loss.detach(), "mlm_loss": mlm_loss.detach(), "po_loss": po_loss.detach()})
        return (outputs, test_batch[6])

    def test_epoch_end(self, outputs):
        if not os.path.isfile("test_embeds_{}_test_run.pkl".format(self.hparams.dataset_name)):
            pkl.dump(outputs, open("test_embeds_{}_test_run.pkl".format(self.hparams.dataset_name), 'wb'))
        else:
            outputs = pkl.load(open("test_embeds_{}_test_run.pkl".format(self.hparams.dataset_name), "rb"))
        #res = faiss.StandardGpuResources()
        if self.hparams.dataset_name != "xlsum":
            self.evaluate_sdr_test_results(outputs)
        else:
            self.evaluate_xlsum_test_results(outputs)


    def evaluate_xlsum_test_results(self, outputs):
        print('test2')
        title = outputs[0][1][0]
        doc_tensors = {}
        doc_tensors[title] = []
        titles = [title]
        for out in outputs:
            # if the next paragraph doesn't belong to the prior document anymore, save it as the first paragraph of
            # the new document
            if out[1][0] != title:
                title = out[1][0]
                titles.append(title)
                doc_tensors[title] = []
            doc_tensors[title].append(out[0].squeeze())
        doc_vectors = self.get_doc_embeddings_avg(doc_tensors)
        vectors = torch.stack(list(doc_vectors.values()))
        # split vectors into articles and summaries
        article_vectors, summary_vectors = vectors[:int(len(vectors)/2)].to(self.device), \
                                           vectors[int(len(vectors)/2):].to(self.device)
        # CAUTION: not the same title ids as in batches
        titles2ids = {k: i for i, k in enumerate(doc_vectors.keys())}
        ids2titles = list(doc_vectors.keys())

        index = faiss.IndexFlatL2(len(summary_vectors[0]))
        if self.device != "cpu":
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        summary_vectors = summary_vectors.to(self.device)
        article_vectors = article_vectors.to(self.device)
        index.add(summary_vectors)
        D, I = index.search(article_vectors, 100)
        I = I + len(article_vectors)
        examples = [[None, title] for title in titles]
        id_examples = [[None, num] for num in range(len(titles))]
        recos = []
        for i in range(len(article_vectors)):
            recos.append((i, I[i]))
        eval_metrics.evaluate_wiki_recos(recos, 'outputs_test', "xlsum", id_examples)


    def get_doc_embeddings_avg(self, outputs):
        """

        :param outputs: Dict of titles with list of respective paragraphs' tensors as values
        :return: averaged tensors for each document
        """
        #avg_tensors = {}
        #for title, tensorlist in outputs.items():
        #    print('yass')
        #    avg_tensors[title] = torch.stack(tensorlist)

        avg_tensors = {title: torch.stack(tensorlist).mean(dim=0) for title, tensorlist in outputs.items()}
        return avg_tensors


    def evaluate_sdr_test_results(self, outputs):
        gt_path = "./data/datasets/{}/gt".format(self.hparams.dataset_name)
        gt = pkl.load(open(gt_path, "rb"))
        doc_embeddings = []
        doc_titles = []
        article2idx = {}
        article2embedding = {}
        flattened_outputs = []
        for i, tup in enumerate(outputs):
            doc_embeddings.append(tup[0].to(self.device))
            doc_titles.extend(tup[1])
            flattened_outputs.extend(list(zip(tup[0], tup[1])))
            for j, title in enumerate(tup[1]):
                article2idx[title] = i * len(tup[1]) + j
                # article2embedding[title] = tup[0][j]

        outputs = flattened_outputs
        doc_embeddings = torch.cat(doc_embeddings, dim=0).cpu()
        index = faiss.IndexFlatL2(len(doc_embeddings[0]))  # build the index
        # index = faiss.index_cpu_to_gpu(res, 0, index)
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
                # gt_as_ids[article] = torch.tensor([doc_titles.index(sim_art) for sim_art in gt[article]])
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


class NPPDataset(LightningDataModule):
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
            collate_fn=partial(NPP_sentence_collate),
        )
        return loader

    def val_dataloader(self):
        sampler = MPerClassSamplerDeter(
            self.val_dataset.labels,
            2,
            length_before_new_iter=self.hparams.limit_val_indices_batches,
            batch_size=self.hparams.val_batch_size,
        )

        loader = DataLoader(
            self.val_dataset,
            num_workers=self.hparams.num_data_workers,
            sampler=sampler,
            batch_size=self.hparams.val_batch_size,
            collate_fn=partial(NPP_sentence_collate),
            shuffle=False,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            num_workers=self.hparams.num_data_workers,
            batch_size=self.hparams.test_batch_size,
            collate_fn=partial(NPP_sentence_collate_test),
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
            #and self.hparams.block_size < self.tokenizer.max_len_single_sentence
            #else self.tokenizer.max_len_single_sentence
            else 512
        )
        if self.dataset_name == "video_games":
            self.train_dataset = WikipediaTextDatasetParagraphOrder(
                tokenizer=self.tokenizer,
                hparams=self.hparams,
                dataset_name=self.dataset_name,
                block_size=block_size,
                mode="train",
            )
            self.val_dataset = WikipediaTextDatasetParagraphOrder(
                tokenizer=self.tokenizer,
                hparams=self.hparams,
                dataset_name=self.dataset_name,
                block_size=block_size,
                mode="val",
            )
            #self.val_dataset.indices_map = self.val_dataset.indices_map[: self.hparams.limit_val_indices_batches]
            #self.val_dataset.labels = self.val_dataset.labels[: self.hparams.limit_val_indices_batches]

            self.test_dataset = XLSumDatasetParagraphOrder(
                tokenizer=self.tokenizer,
                hparams=self.hparams,
                dataset_name=self.dataset_name,
                block_size=block_size,
                mode="test",
            )
        elif self.dataset_name == "xlsum":
            self.train_dataset = XLSumDatasetParagraphOrder(
                tokenizer=self.tokenizer,
                hparams=self.hparams,
                dataset_name=self.dataset_name,
                block_size=block_size,
                mode="train",
            )
            self.val_dataset = XLSumDatasetParagraphOrderTest(
                tokenizer=self.tokenizer,
                hparams=self.hparams,
                dataset_name=self.dataset_name,
                block_size=block_size,
                mode="val",
            )
            # self.val_dataset.indices_map = self.val_dataset.indices_map[: self.hparams.limit_val_indices_batches]
            # self.val_dataset.labels = self.val_dataset.labels[: self.hparams.limit_val_indices_batches]

            self.test_dataset = XLSumDatasetParagraphOrderTest(
                tokenizer=self.tokenizer,
                hparams=self.hparams,
                dataset_name=self.dataset_name,
                block_size=block_size,
                mode="test",
            )
        else:
            raise Exception("Wrong dataset name!")
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
        pe = pe.unsqueeze(1)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class ContrastiveTransformations(object):

    def __init__(self, base_transforms='paragraph_insert', n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]

    def _shuffle_paragraphs(self):
        print('here')
