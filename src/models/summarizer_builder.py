import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from transformers import LongformerModel, BertModel, LongformerConfig

from models.encoder_simple import ExtTransformerEncoder
from models.loss import TripletLoss


class Bert(nn.Module):
    def __init__(self, large, model_name, temp_dir, finetune=False):
        super(Bert, self).__init__()

        if model_name == 'bert':
            if (large):
                self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
            else:
                self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
                # config = BertConfig.from_pretrained('allenai/scibert_scivocab_uncased')
                # config.gradient_checkpointing = True
                # self.model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=temp_dir, config=config)
                # self.model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=temp_dir)

        elif model_name == 'scibert':
            self.model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=temp_dir)

        elif model_name == 'longformer':
            if large:
                self.model = LongformerModel.from_pretrained('allenai/longformer-base-4096', cache_dir=temp_dir)
                # self.model.config.gradient_checkpointing = True
                # self.model = LongformerModel.from_pretrained('allenai/longformer-base-4096', config=self.model.config)
            else:
                self.model = LongformerModel.from_pretrained('allenai/longformer-base-4096', cache_dir=temp_dir)

                # configuration = LongformerConfig.from_pretrained('allenai/longformer-base-4096')
                # configuration.gradient_checkpointing = True
                # self.model = LongformerModel.from_pretrained('allenai/longformer-base-4096', config=configuration)

        self.model_name = model_name
        self.finetune = finetune

    def forward(self, x, mask_src, clss, segs=None):
        if (self.finetune):
            if self.model_name == 'bert' or self.model_name == 'scibert':
                top_vec = self.model(x, attention_mask=mask_src, token_type_ids=segs)['last_hidden_state']
            elif self.model_name == 'longformer':
                global_mask = torch.zeros(mask_src.shape, dtype=torch.long, device='cuda').unsqueeze(0)
                global_mask[:, :, clss.long()] = 1
                global_mask = global_mask.squeeze(0)

                top_vec = self.model(x, attention_mask=mask_src.long(), global_attention_mask=global_mask)['last_hidden_state']

        else:
            self.eval()
            with torch.no_grad():
                if self.model_name == 'bert' or self.model_name == 'scibert':
                    top_vec, _ = self.model(x, attention_mask=mask_src, token_type_ids=segs)

                elif self.model_name == 'longformer':
                    global_mask = torch.zeros(mask_src.shape, dtype=torch.long, device='cuda').unsqueeze(0)
                    global_mask[:, :, clss.long()] = 1
                    global_mask = global_mask.squeeze(0)
                    top_vec = self.model(x, attention_mask=mask_src.long(), global_attention_mask=global_mask)[
                        'last_hidden_state']

        return top_vec


class SentenceExtLayer(nn.Module):
    def __init__(self, in_features=768, out_features=1):
        super(SentenceExtLayer, self).__init__()
        # self.combiner = nn.Linear(1536, 768)
        self.o_feature = out_features
        self.wo = nn.Linear(in_features, out_features, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        # sent_scores = self.seq_model(x)

        # sent_scores = self.sigmoid(self.wo(self.combiner(x)))
        sent_scores = self.sigmoid(self.wo(x))
        # modules = [module for k, module in self.seq_model._modules.items()]
        # input_var = torch.autograd.Variable(x, requires_grad=True)
        # sent_scores = checkpoint_sequential(modules, 2, input_var)
        if self.o_feature > 1:
            sent_scores = sent_scores.squeeze(-1) * mask[:,:,None].repeat(1,1,2).float()
        else:
            sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores


class BertSumEncoder(nn.Module):
    def __init__(self, args, use_transformers=True):
        super(BertSumEncoder, self).__init__()
        self.args = args
        self.use_transformers = use_transformers
        self.bert = Bert(args.large, args.model_name, args.temp_dir, args.finetune_bert)
        if use_transformers:
            self.ext_transformer_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size,
                                                               args.ext_heads,
                                                               args.ext_dropout, args.ext_layers)

        if args.max_pos > 512 and args.model_name == 'bert':
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,
                                                  :].repeat(args.max_pos - 512, 1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        if args.max_pos > 4096 and args.model_name == 'longformer':
            my_pos_embeddings = nn.Embedding(args.max_pos + 2, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:4097] = self.bert.model.embeddings.position_embeddings.weight.data[:-1]
            my_pos_embeddings.weight.data[4097:] = self.bert.model.embeddings.position_embeddings.weight.data[
                                                   1:args.max_pos + 2 - 4096]
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        self.sigmoid = nn.Sigmoid()

    def pooling(self, top_vec, clss, mask_cls):
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss.long()]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        return sents_vec

    def padded(self, input, src):
        target = torch.zeros(src.size(0), src.size(1), dtype=torch.long)
        target[:, :input.size(1)] = input
        return target.to("cuda")

    def forward(self, src, clss, clss_low, clss_tgt, clss_pos, mask_src, src_mask_cls, mask_cls_low, mask_cls_tgt, mask_cls_pos,
                positive_gold=None, positive_sents=None, positive_gold_mask=None, positive_sents_mask=None, negative=None, negative_mask=None, i=None, edge_w=None):


        # # src
        # pos_g = self.padded(positive_gold, src)
        # pos_i = self.padded(positive_intro, src)
        # neg = self.padded(negative, src)
        # # src_all = torch.cat([src, pos_g, pos_i, neg], dim=0)
        # src_all = torch.cat([src, pos_g, neg], dim=0)
        #
        # # mask
        # mask_pos_g = self.padded(positive_gold_mask, mask_src)
        # mask_pos_i = self.padded(positive_intro_mask, mask_src)
        # mask_neg = self.padded(negative_mask, mask_src)
        # # mask_all = torch.cat([mask_src, mask_pos_g, mask_pos_i, mask_neg], dim=0)
        # mask_all = torch.cat([mask_src, mask_pos_g, mask_neg], dim=0)
        #
        # # clss
        #
        # clss_pos1 = self.padded(torch.tensor([[0]]), clss)
        # clss_pos2 = self.padded(torch.tensor([[0]]), clss)
        # clss_neg = self.padded(torch.tensor([[0]]), clss)
        # # clss_all = torch.cat([clss, clss_pos1, clss_pos2, clss_neg], dim=0)
        # clss_all = torch.cat([clss, clss_pos1, clss_neg], dim=0)
        # top_vec = self.bert(src_all, mask_all, clss_all)
        # encoded_sent = self.pooling(top_vec[0,:,:][None, :, :], clss, src_mask_cls)
        # encoded_pos_1 = self.pooling(top_vec[1,:,:][None, :, :], torch.tensor([[0]]).to("cuda"), torch.tensor([[1]]).to("cuda"))
        # # encoded_pos_2 = self.pooling(top_vec[2,:,:][None, :, :], torch.tensor([[0]]).to("cuda"), torch.tensor([[1]]).to("cuda"))
        # encoded_neg_vec = self.pooling(top_vec[3,:,:][None, :, :], torch.tensor([[0]]).to("cuda"), torch.tensor([[1]]).to("cuda"))

        ##### Regular
        # BATCH_SIZE = src.size(0)
        # try:
        # NEG_SAMPLE_SIZE = negative.size(1)
        # NEG_SAMPLE_LEN = negative.size(2)
        # ENCODED_DIMENSION = 768
        encoded_sent = self.bert(src, mask_src, clss)
        # encoded_positive_gold = self.bert(positive_gold, positive_gold_mask, clss_tgt)
        # encoded_neg_vec = self.bert(negative.view(-1,negative.size(-1)), negative_mask.view(-1,negative_mask.size(-1)), clss_low.squeeze(0).view(-1,clss_low.size(-1)))
            # .view(BATCH_SIZE, NEG_SAMPLE_SIZE, NEG_SAMPLE_LEN, ENCODED_DIMENSION)
        # encoded_pos_vec = self.bert(positive_sents, positive_sents_mask, clss_pos)

        encoded_sent = self.pooling(encoded_sent, clss, src_mask_cls)
        # encoded_positive_gold = self.pooling(encoded_positive_gold, clss_tgt, mask_cls_tgt)
        # encoded_positive_sents = self.pooling(encoded_pos_vec, clss_pos, mask_cls_pos)

        # encoded_neg_vec = self.pooling(encoded_neg_vec, clss_low.view(-1,clss_low.size(-1)), mask_cls_low.view(-1,mask_cls_low.size(-1)))\
            # .view(BATCH_SIZE, NEG_SAMPLE_SIZE, clss_low.size(-1), ENCODED_DIMENSION)

        # BERT


        if self.use_transformers:
            encoded_sent = self.ext_transformer_layer(encoded_sent, src_mask_cls, edge_w)
            # encoder_positive_intro = self.ext_transformer_layer(encoder_positive_intro, torch.tensor([[True]]).to("cuda"))
            # encoded_positive_gold = self.ext_transformer_layer(encoded_positive_gold, mask_cls_tgt)
            # encoded_neg_vec = self.ext_transformer_layer(encoded_neg_vec, mask_cls_low)
            # encoded_pos_vec = self.ext_transformer_layer(encoded_positive_sents, mask_cls_pos)

            # averaging positive and negative vectors

            # encoded_positive_gold = torch.mean(encoded_positive_gold, dim=1).unsqueeze(1)
            # encoded_neg_vec = torch.mean(encoded_neg_vec, dim=1).unsqueeze(1)

        # return encoded_sent, encoded_pos_1, encoded_pos_2, encoded_neg_vec
        # return encoded_sent, encoded_pos_vec, encoded_positive_gold, encoded_neg_vec
        return encoded_sent, None, None, None


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.sentence_encoder = BertSumEncoder(args)
        self.sentence_selector = SentenceExtLayer()
        self.loss_sentence_selector = torch.nn.BCELoss(reduction='none')
        self.tr_loss = TripletLoss()
        # self.super_loss = SuperLoss()

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)

        else:
            for p in self.sentence_selector.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        self.to(device)

    def pooling(self, top_vec, clss, mask_cls):
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss.long()]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        return sents_vec

    def forward(self, src, intro_summary, tgt, low_sents, pos_sents, src_sents_rg, src_sents_rg_intro,
                segs, segs_intro, clss, clss_low, clss_pos, clss_tgt, mask_src, mask_intro_summary, mask_tgt, mask_cls, mask_cls_low, mask_cls_pos, mask_cls_tgt,
                mask_low_sents, mask_pos_sents, sent_bin_labels, p_id, is_inference=False, i=None, return_encodings=False, edge_w=None):

        # try:
        source_sents_repr, pos_oracle_repr, pos_gold_repr, neg_repr = self.sentence_encoder(src, clss, clss_low, clss_tgt, clss_pos,  mask_src, mask_cls, mask_cls_low, mask_cls_tgt, mask_cls_pos,
                                                                            positive_gold=tgt,
                                                                            positive_sents=pos_sents,
                                                                            positive_gold_mask=mask_tgt,
                                                                            positive_sents_mask=mask_pos_sents,
                                                                            negative=low_sents,
                                                                            negative_mask=mask_low_sents,
                                                                            i = i,
                                                                            edge_w=edge_w,
                                                        )
        # except Exception as e:
        #     print(e)
        #     import pdb;pdb.set_trace()

        # Triplet Loss
        # tr_loss, distance_neg, distance_pos = self.tr_loss(source_sents_repr, rep_pos_oracle=pos_oracle_repr, rep_pos_gold=pos_gold_repr,
        #                                                    rep_negative=neg_repr, src_labels=sent_bin_labels, anchor_mask=mask_cls,
        #                                                    oracle_pos_mask=mask_cls_pos, negative_mask=mask_low_sents, gold_mask=mask_cls_tgt, mask_cls_low= mask_cls_low)
        # tr_loss = (tr_loss * mask_cls.float())


        #### now let's do sentence selector loss!!

        src_sents_score = self.sentence_selector(source_sents_repr, mask_cls)
        loss_sent = self.loss_sentence_selector(src_sents_score, sent_bin_labels.float())
        # loss_sent = self.super_loss(src_sents_score, sent_bin_labels.float())
        loss_sent = (loss_sent * mask_cls.float())
        # loss = loss_sent
        # adding up losses
        # loss = (.99) * tr_loss + (.01) * loss_sent
        loss = loss_sent
        # rg_scores =  self.rg_predictor(source_sents_repr, mask_cls)
        # loss_rg = self.loss_rg_predictor(rg_scores, torch.cat((src_sents_rg[:,:,None], src_sents_rg_intro[:,:,None]), dim=2).float())
        # loss_rg = (loss_rg * mask_cls[: ,:, None].repeat(1,1,2).float()).sum(dim=1).sum()

        if return_encodings:
            return src_sents_score, mask_cls, loss, None, None, source_sents_repr

        # return src_sents_score, mask_cls, loss.sum(), tr_loss.sum(), loss_sent.sum(), distance_neg, distance_pos
        return src_sents_score, mask_cls, loss.sum(), torch.tensor([0]), loss_sent.sum(), None, None
        # return src_sents_score, mask_cls, loss.sum(), torch.tensor([0]), torch.tensor([0]), None, None
