from dataclasses import dataclass

from transformers.models.t5.modeling_t5 import (
    T5Stack, T5Block, T5LayerNorm, T5LayerSelfAttention, T5LayerFF, T5LayerCrossAttention,
    T5PreTrainedModel, T5ForConditionalGeneration
)
# from transformers.models.t5.modeling_t5_prompt_v2 import T5Block_prompt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import copy

from transformers.modeling_outputs import ModelOutput, BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from transformers import BeamScorer, BeamSearchScorer


logger = logging.get_logger(__name__)


class VisualEmbedding(nn.Module):
    def __init__(self, config, obj_order_embedding):
        super().__init__()
        self.config = config
        feat_dim = config.feat_dim
        pos_dim = config.pos_dim
        # n_objs = config.n_objs
        n_images = config.n_images

        if self.config.individual_vis_layer_norm:

            # Object feature encoding
            feat_embedding = [nn.Linear(feat_dim, config.d_model)]
            if self.config.use_vis_layer_norm:
                feat_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            self.feat_embedding = nn.Sequential(*feat_embedding)

            # self.relative_vis_pos_embedding = nn.Linear(pos_dim + 1, config.num_heads)
            absolute_vis_pos_embedding = [nn.Linear(pos_dim + 1, config.d_model)]
            if self.config.use_vis_layer_norm:
                absolute_vis_pos_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            self.absolute_vis_pos_embedding = nn.Sequential(*absolute_vis_pos_embedding)
            # self.absolute_vis_pos_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

            if self.config.use_vis_order_embedding:
                # self.obj_order_embedding = nn.Embedding(n_objs, config.d_model)
                self.obj_order_embedding = obj_order_embedding
                self.img_order_embedding = nn.Embedding(n_images, config.d_model)

        else:
            # Object feature encoding
            feat_embedding = [nn.Linear(feat_dim, config.d_model)]
            # if self.config.use_vis_layer_norm:
            #     feat_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            self.feat_embedding = nn.Sequential(*feat_embedding)

            # self.relative_vis_pos_embedding = nn.Linear(pos_dim + 1, config.num_heads)
            absolute_vis_pos_embedding = [nn.Linear(pos_dim + 1, config.d_model)]
            # if self.config.use_vis_layer_norm:
            #     absolute_vis_pos_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            self.absolute_vis_pos_embedding = nn.Sequential(*absolute_vis_pos_embedding)
            # self.absolute_vis_pos_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

            if self.config.use_vis_order_embedding:
                # self.obj_order_embedding = nn.Embedding(n_objs, config.d_model)
                self.obj_order_embedding = obj_order_embedding
                self.img_order_embedding = nn.Embedding(n_images, config.d_model)

            if self.config.use_vis_layer_norm:
                self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def get_area(self, pos):
        """
        Args
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            area : [B, N]
        """
        # [B, N]
        height = pos[:, :, 3] - pos[:, :, 2]
        width = pos[:, :, 1] - pos[:, :, 0]
        area = height * width
        return area


    def forward(self, feats, pos, img_order_ids=None, obj_order_ids=None):
        """
        Args
            feats: [B, N, feat_dim]
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            relative_vis_pos_embedding: [B, N, N, n_heads]
            absolute_vis_pos_embedding: # [B, N, d_model]
        """

        B, N, _ = feats.size()
        assert pos.size() == (B, N, 4)

        feat_embedding = self.feat_embedding(feats)

        device = feats.device
        dtype = feats.dtype

        area = self.get_area(pos).unsqueeze(2) # [B, N, 1]
        pos = torch.cat([pos, area], dim=2) # [B, N, 5]

        # [B, N, d_model]
        absolute_vis_pos_embedding = self.absolute_vis_pos_embedding(pos)
        # absolute_vis_pos_embedding = self.absolute_vis_pos_layer_norm(absolute_vis_pos_embedding)


        if self.config.use_vis_order_embedding:
            if img_order_ids is None:
                img_order_ids = torch.zeros(N, dtype=torch.long, device=device)
                img_order_ids = img_order_ids.unsqueeze(0) #.expand(B, -1)
            img_order_embedding = self.img_order_embedding(img_order_ids)

            if obj_order_ids is None:
                obj_order_ids = torch.arange(N, dtype=torch.long, device=device)
                obj_order_ids = obj_order_ids.unsqueeze(0) #.expand(B,-1)
            # assert obj_order_ids.max().item() < 32200, obj_order_ids
            obj_order_ids = self.obj_order_embedding.num_embeddings - obj_order_ids - 1
            obj_order_embedding = self.obj_order_embedding(obj_order_ids)

            vis_embedding = feat_embedding + absolute_vis_pos_embedding + \
                img_order_embedding + obj_order_embedding

        else:
            vis_embedding = feat_embedding + absolute_vis_pos_embedding

        if not self.config.individual_vis_layer_norm:
            if self.config.use_vis_layer_norm:
                vis_embedding = self.layer_norm(vis_embedding)

        return vis_embedding

class JointEncoder(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super(T5Stack, self).__init__(config)
        self.config = config

        self.embed_tokens = embed_tokens
        self.is_decoder = self.config.is_decoder
        assert self.config.is_decoder is False

        self.visual_embedding = VisualEmbedding(self.config, embed_tokens)

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=(i == 0))
                for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()
        self.model_parallel = False
        self.device_map = None
        print("========== Original Joint Encoder ========== ")


    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings
        self.visual_embedding.obj_order_embedding = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,

        vis_inputs=None,
        vis_attention_mask=None,

        # task_id=None,

        inputs_embeds=None,
        head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        if inputs_embeds is None: #  文本
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        B, L = inputs_embeds.size()[:-1]

        vis_feats = vis_inputs[0]
        boxes = vis_inputs[1]
        img_order_ids = None
        obj_order_ids = None
        if len(vis_inputs) >= 3:
            img_order_ids = vis_inputs[2]
        if len(vis_inputs) == 4:
            obj_order_ids = vis_inputs[3]

        vis_embeds = self.visual_embedding(
            vis_feats, boxes, img_order_ids, obj_order_ids)

        V_L = vis_embeds.size(1)

        inputs_embeds = torch.cat([inputs_embeds, vis_embeds], dim=1)


        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

            # Generate mask ------------------------------
            # if self.prefix is not None and self.batched_prompt is not None:
            #     prefix_len = prefix_layer.shape[-3]  # 1+5

            if attention_mask is None:
                attention_mask = input_ids.ne(self.config.pad_token_id).to(dtype=inputs_embeds.dtype,
                                                                           device=inputs_embeds.device)

            if vis_attention_mask is None:
                vis_attention_mask = attention_mask.new_ones(B, V_L)

            attention_mask = torch.cat([attention_mask, vis_attention_mask], dim=1)

            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, (B, L + V_L),
                                                                       inputs_embeds.device)  # [bs, 1, 1, 56+6]

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        # position_bias = None
        # encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        if self.config.num_layers > 0:
            assert self.block[0].layer[0].SelfAttention.has_relative_attention_bias

            seq_length = L + V_L

            q_len = seq_length  # 56 + x
            k_len = seq_length  # 56 + x

            # [1, n_heads, Q_len, K_len]
            text_position_bias = self.block[0].layer[0].SelfAttention.compute_bias(L, L)
            num_heads = text_position_bias.size(1)
            position_bias = text_position_bias.new_zeros(
                1, num_heads, seq_length, seq_length)  # [1, 12, 56, 56]
            position_bias[:, :, :L, :L] = text_position_bias

            # print('position_bias size', position_bias.size())
            # print('attention_mask size', attention_mask.size())
            # print('extended_attention_mask size', extended_attention_mask.size())
            # relative position bias only between Text <-> Text
            # no relative position bias Text -> Vision
            # no relative position bias Vision -> Text
            # no relative position bias Vision <-> Vision
            # position_bias[:, :, L:, :] = 0
            # position_bias[:, :, :, L:] = 0
            position_bias = position_bias + extended_attention_mask

            for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
                # print('Layer...', i)

                # if output_hidden_states:
                #     all_hidden_states = all_hidden_states + (hidden_states,)

                # print("Prompt shape...",prefix_layer.shape)
                layer_outputs = layer_module(
                    hidden_states, # hidden_states = self.dropout(inputs_embeds)
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    encoder_decoder_position_bias=None,
                    head_mask=head_mask[i],
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                # layer_outputs is a tuple with:
                # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                hidden_states, present_key_value_state = layer_outputs[:2]

                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention weights),
                # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                position_bias = layer_outputs[2]

                # append next layer key value states
                if use_cache:
                    present_key_value_states = present_key_value_states + \
                        (present_key_value_state,)

                # if output_attentions:
                #     all_attentions = all_attentions + (layer_outputs[3],)
                #     if self.is_decoder:
                #         all_cross_attentions = all_cross_attentions + \
                #             (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class VLT5(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super(T5ForConditionalGeneration, self).__init__(config)

        self.config = config

        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        #---- Modified ----#
        # self.encoder = T5Stack(encoder_config, self.shared)
        self.encoder = JointEncoder(encoder_config, self.shared)
        #------------------#

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True # decoder
        decoder_config.is_encoder_decoder = False

        self.decoder = T5Stack(decoder_config, self.shared) # decoder

        #  -------- there, vocab_size可以扩大吗
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.prototype_fc1 = nn.Linear(config.d_model, config.d_model)
        self.prototype_fc2 = nn.Linear(config.d_model, config.d_model)
        self.L = 20
        self.V_L = 36


        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        # self.Q_prototype_num = torch.zeros(10) # 各个类别的计算过的数量
        # self.V_prototype_num = torch.zeros(80)
        # self.Q_prototype = None
        # self.V_prototype = None

        self.Q_task_mem_proto = {}  # 当前时间下，memory类别对应的prototype
        self.V_task_mem_proto = {}
        self.Q_task_cur_proto = {}  # 当前时间下，memory类别对应的prototype
        self.V_task_cur_proto = {}
        self.Q_prototype_num = {}
        self.V_prototype_num = {}
        print("Q_task_mem_proto and Q_task_cur_proto")

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def extend_vocab(self, vocab_size):

        new_shared = nn.Embedding(vocab_size, self.config.d_model)
        old_weight = self.shared.weight.data.detach().clone()
        old_vocab_size = old_weight.size(0)
        new_shared.weight.data[:old_vocab_size, :] = old_weight
        self.shared = new_shared

        new_lm_head = nn.Linear(self.config.d_model, vocab_size, bias=False)
        old_weight = self.lm_head.weight.data.detach().clone()
        old_vocab_size = old_weight.size(0)
        new_lm_head.weight.data[:old_vocab_size, :] = old_weight
        self.lm_head = new_lm_head

        self.vis_encoder.visual_embedding.obj_order_embedding = self.shared

        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

        self.lm_head.weight = self.shared.weight

        self.config.vocab_size = vocab_size
        self.encoder.config.vocab_size = vocab_size
        self.vis_encoder.config.vocab_size = vocab_size
        self.decoder.config.vocab_size = vocab_size




    # @add_start_docstrings_to_callable(T5_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)

    def cosine_similarity_multi(self, a, b, labels=None, rep="real"):
        """
        Compute the cosine similarity between two vectors

        Parameters:
        ----------
        a:  Tensor(N_a,D)
        b:  Tensor(N_b,D)
        rep: str
            Representation to compute cosine similarity: real | bipolar | tanh
        Return
        ------
        similarity: Tensor(N_a,N_b)
        """
        sim_act = nn.Tanh()  # meta-train: tanh
        a_normalized = F.normalize(sim_act(a), dim=1) # 这里的维度是不对的 # [class_num, 768]
        b_normalized = F.normalize(sim_act(b), dim=1) #[bs, dim]
        similiarity = F.linear(a_normalized, b_normalized).transpose(1,0) # [bs, class_num]
        # similarities_sharpened = sharpening_activation(similarity)
        max_idx = torch.argmax(similiarity, dim=1) #[bs]
        selected_prototype = a[max_idx] # [bs, 768]

        if labels is not None:
            labels = torch.topk(labels, 1)[1].squeeze(1) # convert one-hot to label
            acc = (max_idx == labels).sum()//labels.shape[0]
            # print("current retreieval acc is:", 100*acc,'%')
        else:
            acc = -1

        return selected_prototype, max_idx, acc

    def update_prototype(self, current_Q_prototype, current_V_prototype, current_num_Q, current_num_V, current_task_id, proto_alpha, proto_beta):

        # if self.Q_prototype == None and self.V_prototype == None:
        #     self.Q_prototype = current_Q_prototype
        #     self.Q_prototype_num = current_num_Q
        #     self.V_prototype = current_V_prototype
        #     self.V_prototype_num = current_num_V
        if current_task_id not in self.Q_task_cur_proto:
            self.Q_task_cur_proto[current_task_id] = current_Q_prototype  # 目的是存当前task的prototype
            self.Q_prototype_num = current_num_Q
            self.V_prototype_num = current_num_V
            self.V_prototype = current_V_prototype # ---- 每个task从新计算
            if current_task_id == 0:
                self.Q_prototype = current_Q_prototype
            else:
                self.Q_prototype[current_task_id] = current_Q_prototype[current_task_id]
        else:
            # 先按均值来
            # 对于current-task-id, 也应该按当前的均值
            # div_Q = current_num_Q.unsqueeze(1) + self.Q_prototype_num.unsqueeze(1)
            # ones = torch.ones((current_num_Q.shape[0], 1)).to(torch.device('cuda'))
            # div_Q = torch.where(div_Q <= 0, ones, div_Q)
            # existed_Q_proto = self.Q_task_cur_proto[current_task_id] * self.Q_prototype_num.unsqueeze(1) # 只取这个数值
            # self.Q_task_cur_proto[current_task_id] = (current_Q_prototype * current_num_Q.unsqueeze(1) + existed_Q_proto.detach()) / div_Q
            # 对于current-task-id, 只用当前batch的均值
            self.Q_task_cur_proto[current_task_id] = current_Q_prototype # trick
            # print('proto_alpha:', proto_alpha)


            # 对于旧任务，应该是当前memory中存的样本的均值
            if current_task_id != 0:
                if current_task_id not in self.Q_task_mem_proto:
                    current_Q_prototype_mem = current_Q_prototype.clone()
                    current_Q_prototype_mem[current_task_id] = 0 # 当前的制0，保留memory
                    self.Q_task_mem_proto[current_task_id] = current_Q_prototype_mem
                else:
                    current_Q_prototype_mem = current_Q_prototype.clone()
                    current_Q_prototype_mem[current_task_id] = 0  # 当前的制0，保留memory
                    self.Q_task_mem_proto[current_task_id] = proto_alpha*self.Q_task_mem_proto[current_task_id] + (1-proto_alpha)*current_Q_prototype_mem.detach() # simply 2

                self.Q_prototype = self.Q_task_mem_proto[current_task_id].detach() # 取旧task的memory-prototype
                self.Q_prototype[current_task_id] = self.Q_task_cur_proto[current_task_id][current_task_id].detach() # 只取当前task的prototype # trick
            else:
                self.Q_prototype = self.Q_task_cur_proto[current_task_id]


            # div_V = current_num_V.unsqueeze(1) + self.V_prototype_num.unsqueeze(1)
            # ones = torch.ones((current_num_V.shape[0], 1)).to(torch.device('cuda'))
            # div_V = torch.where(div_V <= 0, ones, div_V)
            # existed_V_proto = self.V_prototype*self.V_prototype_num.unsqueeze(1)
            # self.V_prototype = (current_V_prototype * current_num_V.unsqueeze(1) + existed_V_proto.detach())/ div_V

            self.V_prototype = proto_beta*self.V_prototype + (1-proto_beta)*current_V_prototype
            # 如果是memory的话，就直接平均吗

            self.Q_prototype_num = self.Q_prototype_num.detach() + current_num_Q
            self.V_prototype_num = self.V_prototype_num.detach() + current_num_V

    def calculate_current_prototype(self, fc_hidden_Q, labels):
        # fc_hidden_Q = hidden_states[:, :self.L, :]
        # fc_hidden_Q = self.prototype_fc1(fc_hidden_Q) # ---- 这里应该是batch的呀，确实是batch  [bs, 20, dim]
        fc_hidden_Q = torch.mean(fc_hidden_Q, dim=1)  # ---- mean-pooling

        div_item_ = torch.sum(labels, dim=0).unsqueeze(1).repeat(1, 768)  # [num_classes1, dim]
        ones = torch.ones((labels.shape[1], fc_hidden_Q.shape[-1])).to(torch.device('cuda'))  # [num_classes1, dim]
        div_item = torch.where(div_item_ <= 0, ones, div_item_)  # 防止除以0

        # 没有考虑mask呢....
        current_prototype_Q = torch.matmul(torch.transpose(labels, 0, 1),
                                           fc_hidden_Q) / div_item  # [num_classes1, dim]

        current_num = torch.sum(labels, dim=0) #[num_classes]
        return current_prototype_Q, current_num

    def memory_loss(self, hidden_Q, hidden_V, ques_labels, cate_labels):
        # loss_d = ((src_centers[prev_idx] - feats_norm[pick]).pow(2).sum(1)).mean()
        hidden_Q_ = torch.mean(hidden_Q, dim=1) # [bs, dim]
        correspond_Q_prototype = torch.matmul(ques_labels, self.Q_prototype)# [10, dim] and [bs,10] -> [bs, dim]
        loss_d_Q = ((hidden_Q_ - correspond_Q_prototype.detach()).pow(2).sum(dim=1)).mean() # *0.1
        # ------ detach() here， 不更新self.prototype -----

        hidden_V_ = torch.mean(hidden_V, dim=1)
        correspond_V_prototype = torch.matmul(cate_labels, self.V_prototype)# [10, dim] and [bs,10] -> [bs, dim]
        loss_d_V = ((hidden_V_ - correspond_V_prototype.detach()).pow(2).sum(dim=1)).mean() # 0.005

        return loss_d_Q, loss_d_V



    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,

        vis_inputs=None,
        vis_attention_mask=None,

        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        reduce_loss=False,

        return_hidden_state=False,

        **kwargs,
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:

            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,

                vis_inputs=vis_inputs,
                vis_attention_mask=vis_attention_mask,

                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(
                    encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(
                    encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0] # [bs, L+V_L, 768]


        # ==================================================================
        if 'cate_labels' in kwargs:
            cate_labels = kwargs['cate_labels'] #[bs, num_classes]
        if 'ques_labels' in kwargs:
            ques_labels = kwargs['ques_labels'] #[bs, num_classes]

        if 'current_task_id' in kwargs:
            current_task_id = kwargs['current_task_id']

        if 'proto_alpha' in kwargs:
            proto_alpha = kwargs['proto_alpha']
        if 'proto_beta' in kwargs:
            proto_beta = kwargs['proto_beta']

        if 'proto_update' in kwargs and kwargs['proto_update']: # only for training

            # 生成当前batch的prototype
            current_prototype_Q, current_num_Q = self.calculate_current_prototype(hidden_states[:, :self.L, :], ques_labels)
            current_prototype_V, current_num_V = self.calculate_current_prototype(hidden_states[:, self.L:, :], cate_labels)

            if 'memory' in kwargs and kwargs['memory']==True:
                loss_memory_Q, loss_memory_V = self.memory_loss(hidden_states[:, :self.L, :], hidden_states[:, self.L:, :], ques_labels, cate_labels)
            else:
                loss_memory_Q, loss_memory_V = 0,0

            # 更新prototype
            self.update_prototype(current_prototype_Q, current_prototype_V, current_num_Q, current_num_V, current_task_id, proto_alpha, proto_beta)

            # 检索找到最相关的
            retrievaled_Q_proto, max_idx_Q, acc_Q = self.cosine_similarity_multi(self.Q_prototype, torch.mean(hidden_states[:, :self.L, :], dim=1), ques_labels) # [bs, 768]
            retrievaled_Q_proto = retrievaled_Q_proto.unsqueeze(1) #[bs, 1, 768]
            retrievaled_V_proto, max_idx_V, acc_V = self.cosine_similarity_multi(self.V_prototype, torch.mean(hidden_states[:, self.L:, :], dim=1), cate_labels) # [bs, 768]
            retrievaled_V_proto = retrievaled_V_proto.unsqueeze(1) #[bs, 1, 768]

        else:
            retrievaled_Q_proto, max_idx_Q, acc_Q = self.cosine_similarity_multi(self.Q_prototype, torch.mean(hidden_states[:, :self.L, :], dim=1))  # [bs, 768]
            retrievaled_Q_proto = retrievaled_Q_proto.unsqueeze(1)  # [bs, 1, 768]
            retrievaled_V_proto, max_idx_V, acc_V = self.cosine_similarity_multi(self.V_prototype, torch.mean(hidden_states[:, self.L:, :], dim=1))  # [bs, 768]
            retrievaled_V_proto = retrievaled_V_proto.unsqueeze(1)  # [bs, 1, 768]
            loss_memory_Q, loss_memory_V = 0, 0

        # 加入计算
        hidden_states = torch.cat((hidden_states, retrievaled_Q_proto.detach(), retrievaled_V_proto.detach()), dim=1)
        # print('detach()')


        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        if attention_mask is None: # ---- 每次都是这里
            attention_mask = input_ids.ne(self.config.pad_token_id).to(dtype=hidden_states.dtype, device=hidden_states.device) #[bs, L]

        if vis_attention_mask is None: # ---- 每次都是这里
            B, L = attention_mask.size()
            # V_L = encoder_outputs[0].size(1) - L # 这里看起来问题不大
            V_L = hidden_states.size(1) - L #
            vis_attention_mask = attention_mask.new_ones(B, V_L) # 全1
        encoder_attention_mask = torch.cat([attention_mask, vis_attention_mask], dim=1) #[bs, V_L]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,

            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,

            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # print('decoder_outputs')
        # print(decoder_outputs)

        sequence_output = decoder_outputs[0]

        assert self.config.tie_word_embeddings is True

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        if return_hidden_state:
            return sequence_output

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss(ignore_index=-100)
            # loss = loss_fct(
            #     lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

            if reduce_loss:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1))

            # print('loss')
            # print(loss)

        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #     return ((loss,) + output) if loss is not None else output

        return VLSeq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            # decoder_attentions=decoder_outputs.attentions,
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            # encoder_attentions=encoder_outputs.attentions,
            # vis_encoder_last_hidden_state=vis_encoder_outputs.last_hidden_state,
            # vis_encoder_hidden_states=vis_encoder_outputs.hidden_states,
            # vis_encoder_attentions=vis_encoder_outputs.attentions,
            # cross_encoder_outputs=cross_encoder_outputs
            loss_memory_Q = loss_memory_Q,
            loss_memory_V = loss_memory_V,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None,
        encoder_outputs=None,
        **kwargs):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        output = {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

        if 'vis_attention_mask' in kwargs:
            output['vis_attention_mask'] = kwargs['vis_attention_mask']

        return output

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1,
                                                                expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx)

        if model_kwargs.get("vis_attention_mask", None) is not None:
            model_kwargs['vis_attention_mask'] = model_kwargs['vis_attention_mask'].index_select(
                0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx
            )
            model_kwargs["encoder_outputs"] = encoder_outputs

        return input_ids, model_kwargs


@dataclass
class VLSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Languaged modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see ``past_key_values`` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    vis_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    vis_encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vis_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    # new_add
    encoder_attention_mask: Optional[Tuple[torch.FloatTensor]] = None
    loss_memory_Q: torch.FloatTensor = None
    loss_memory_V: torch.FloatTensor = None

    # cross_encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None
