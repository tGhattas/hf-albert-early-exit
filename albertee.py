import numpy as np
from transformers import AlbertForSequenceClassification, AlbertConfig, AlbertModel
from transformers.models.albert.modeling_albert import AlbertTransformer, AlbertModel, BaseModelOutput, \
    AlbertLayerGroup, \
    SequenceClassifierOutput, AlbertPreTrainedModel, CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, AlbertEmbeddings, \
    BaseModelOutputWithPooling
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from dataclasses import dataclass
from torch import nn
from torch.nn.functional import softmax
from typing import Dict, List, Optional, Tuple, Union
import torch


class ExitLayer(nn.Module):
    """
    output:(logits,)

    """

    def __init__(self, config: AlbertConfig):
        super().__init__()
        self.config = config
        self.exit_thres = config.exit_thres
        self.num_labels = config.num_labels

        if not config.use_out_pooler:
            self.pooler = nn.Linear(config.hidden_size, config.fc_size1)
        self.pooler_activation = nn.Tanh()
        # classifier
        self.dropout = nn.Dropout(config.classifier_dropout_prob)

        self.classifier = nn.Linear(config.fc_size1, config.num_labels)

        # statistics
        self.exit_loss = 0
        self.entropy = 0
        self.softm_logits = 0

        # exit
        # ori: original exit, entropy: exit by entropy
        self.exit_cnt_dict = {"entropy": 0, "ori": 0}
        self.is_right = False

    def forward(self, encoder_outputs, pooler=None):
        sequence_output = encoder_outputs[0]
        if self.config.pooler_input == "cls":
            pool_input = sequence_output[:, 0]
        elif self.config.pooler_input == "cls-mean":
            t1 = sequence_output[:, 0]
            t2 = sequence_output.mean(dim=1)
            pool_input = t1 + t2
        elif self.config.pooler_input == "cls-max":
            t1 = sequence_output[:, 0]
            t2 = torch.max(sequence_output, dim=1)[0]
            pool_input = t1 + t2
        elif self.config.pooler_input == "mean":
            pool_input = sequence_output.mean(dim=1)
        elif self.config.pooler_input == "max":
            pool_input = torch.max(sequence_output, dim=1)[0]
        elif self.config.pooler_input == "mean-max":
            t1 = sequence_output.mean(dim=1)
            t2 = torch.max(sequence_output, dim=1)[0]
            pool_input = t1 + t2
        else:
            raise Exception("Wrong pooler input!")

        if self.config.use_out_pooler:
            pooled_output = self.pooler_activation(pooler(pool_input))
        else:
            pooled_output = self.pooler_activation(self.pooler(pool_input))

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,)  # + outputs[2:]

        tmp = softmax(logits, dim=1)
        self.softm_logits = tmp
        self.entropy = (tmp * torch.log(tmp)).sum(dim=1) / \
                       np.log(1 / self.num_labels)

        return outputs  # (logits,)

    def label_smoothing(self, x):
        e = self.config.e_label_smoothing
        return ((1 - e) * x) + (e / x.shape[-1])

    def lower_than_thres(self, x=None):
        if self.config.thres_name == "entropy":
            # normalized entropy
            if self.entropy.mean() < self.exit_thres:
                self.exit_cnt_dict["entropy"] = self.exit_cnt_dict["entropy"] + 1
                return True
            else:
                self.exit_cnt_dict["ori"] = self.exit_cnt_dict["ori"] + 1
                return False
        elif self.config.thres_name == "logits":
            # logits on both sides?
            return self.softm_logits.min() < self.exit_thres
        elif self.config.thres_name == "bias_1":
            if self.entropy.mean() < self.exit_thres:
                self.exit_cnt_dict["entropy"] = self.exit_cnt_dict["entropy"] + 1
                return True
            cnt_bias = int(self.config.cnt_thres)
            margin = self.config.margin
            if len(x) < cnt_bias:
                self.exit_cnt_dict["ori"] = self.exit_cnt_dict["ori"] + 1
                return False
            cnt_large = 0
            cnt_less = 0

            for i in range(cnt_bias, 0, -1):
                if x[-i] < 0.5 - margin:
                    cnt_less += 1
                elif x[-i] > 0.5 + margin:
                    cnt_large += 1
            if cnt_large == cnt_bias or cnt_less == cnt_bias:
                self.exit_cnt_dict["bias_1"] = self.exit_cnt_dict.get(
                    "bias_1", 0) + 1
                return True
            else:
                self.exit_cnt_dict["ori"] = self.exit_cnt_dict["ori"] + 1
                return False


@dataclass
class BaseModelOutputWithPoolingEarlyExit(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    exits_idx: Optional[int] = None
    exits_logits: Optional[torch.FloatTensor] = None

@dataclass
class BaseModelOutputWithEarlyExit(ModelOutput):
    """
        Base class for model's outputs, with potential hidden states and attentions.

        Args:
            last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the model.
            hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
                Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
                one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
            attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
                Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
                sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
        """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    exits_idx: Optional[int] = None
    exits_logits: Optional[torch.FloatTensor] = None


class AlbertTransformerEarlyExit(nn.Module):

    def __init__(self, config: AlbertConfig):
        super().__init__()

        self.config = config
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])

        if config.num_exit_layers == 1:
            self.exit_out_layer = ExitLayer(config)
        else:
            self.exit_out_layers = nn.ModuleList([ExitLayer(config) for _ in range(config.num_exit_layers)])

        self.exit_logits_lst = []
        self.cnt_sp = 0

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
            pooler: Optional[nn.Module] = None,
            is_eval_mode: bool = False,
    ) -> Union[BaseModelOutputWithEarlyExit, Tuple]:
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        self.exit_logits_lst = []
        exit_sm_logits_lst = []

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        head_mask = [None] * self.config.num_hidden_layers if head_mask is None else head_mask

        count = 0
        exit_logits: Optional[torch.FloatTensor] = None
        while count < self.config.num_hidden_layers:
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(count / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[group_idx * layers_per_group: (group_idx + 1) * layers_per_group],
                output_attentions,
                output_hidden_states,
            )
            hidden_states = layer_group_output[0]

            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.config.num_exit_layers == 1:
                exit_out_layer = self.exit_out_layer
            else:
                layer_idx = count // (self.config.num_hidden_layers //
                                      self.config.num_exit_layers)
                exit_out_layer = self.exit_out_layers[layer_idx]

            # logits before softmax
            exit_logits = exit_out_layer((hidden_states,), pooler=pooler)
            self.exit_logits_lst.append(exit_logits)
            count += 1
            if is_eval_mode:
                # only exit early during inference
                sm_logits = softmax(exit_logits[0], dim=1)
                exit_sm_logits_lst.append(sm_logits[0][0])
                if exit_out_layer.lower_than_thres(exit_sm_logits_lst):
                    break

        self.cnt_sp += 1

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, count - 1, exit_logits] if v is not None)

        assert exit_logits is not None
        return BaseModelOutputWithEarlyExit(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            exits_idx=count - 1,
            exits_logits=exit_logits
        )


########################################################################################################################
class AlbertModelEarlyExit(AlbertPreTrainedModel):
    config_class = AlbertConfig
    base_model_prefix = "albert"

    def __init__(self, config: AlbertConfig):
        super().__init__(config)

        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformerEarlyExit(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh() # TODO: check if this is necessary


        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} ALBERT has
        a different architecture in that its layers are shared across groups, which then has inner groups. If an ALBERT
        model has 12 hidden layers and 2 hidden groups, with two inner groups, there is a total of 4 different layers.

        These layers are flattened: the indices [0,1] correspond to the two inner groups of the first hidden layer,
        while [2,3] correspond to the two inner groups of the second hidden layer.

        Any layer with in index other than [0,1,2,3] will result in an error. See base class PreTrainedModel for more
        information about head pruning
        """
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            inner_group_idx = int(layer - group_idx * self.config.inner_group_num)
            self.encoder.albert_layer_groups[group_idx].albert_layers[inner_group_idx].attention.prune_heads(heads)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            is_eval_mode: bool = False,
    ) -> Union[BaseModelOutputWithPoolingEarlyExit, Tuple]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_eval_mode=is_eval_mode,
            pooler=self.pooler
        )

        sequence_output = encoder_outputs[0]
        exit_idx = encoder_outputs[-2]
        exits_logits = encoder_outputs[-1]


        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0])) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingEarlyExit(  # TODO: note the pooling
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            exits_idx = exit_idx,
            exits_logits=exits_logits
        )


########################################################################################################################
class AlbertForSequenceClassificationEarlyExit(AlbertPreTrainedModel):
    def __init__(self, config: AlbertConfig, num_exit_layers: int, exit_thres: float, use_out_pooler: bool, fc_size1: int, pooler_input: str, w_init: float, weight_name: str):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.config.num_exit_layers = num_exit_layers
        self.config.exit_thres = exit_thres
        self.config.use_out_pooler = use_out_pooler
        self.config.fc_size1 = fc_size1
        self.config.pooler_input = pooler_input
        self.config.w_init = w_init
        self.config.weight_name = weight_name
        self.data_num = self.config.num_exit_layers + 1

        self.albert = AlbertModelEarlyExit(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.exit_weight_lst = None
        self.ori_weight = None
        if self.config.weight_name == "dyn":
            self.sigmoid = nn.Sigmoid()
            self.W = torch.tensor([self.config.w_init for _ in range(self.config.num_exit_layers)],
                                  device="cuda" if torch.cuda.is_available() else "cpu",
                                  requires_grad=True)
            self.M = self.config.num_exit_layers + 1

        elif self.config.weight_name == "equal":
            self.ori_weight = 1.
            self.exit_weight_lst = [1.] * self.config.num_exit_layers

        # Initialize weights and apply final processing
        self.post_init()

        self.cnt_exit = [0] * self.data_num
        self.compute_ratio = 0

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            is_eval_mode: bool = False,
    ) -> Union[SequenceClassifierOutput, Tuple]:
        """ early exit implementation """
        r"""
                labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                    Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                    config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                    `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
                """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_eval_mode=is_eval_mode
        )

        loss = None
        result = None
        if not is_eval_mode:

            if self.config.weight_name == "dyn":
                self.exit_weight_lst = self.sigmoid(self.W)
                self.ori_weight = self.M - self.exit_weight_lst.sum()

            exit_logits_lst = self.albert.encoder.exit_logits_lst
            loss_weighted = None
            loss_all = []
            for i, logits in enumerate(exit_logits_lst):
                logits = logits[0]  # tuple to tensor
                if labels is not None:
                    # set problem type
                    if self.config.problem_type is None:
                        if self.num_labels == 1:
                            self.config.problem_type = "regression"
                        elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                            self.config.problem_type = "single_label_classification"
                        else:
                            self.config.problem_type = "multi_label_classification"

                    # print(self.config.problem_type, 'problem type') TODO: remove
                    if self.config.problem_type == "regression":
                        loss_fct = MSELoss()
                        if self.num_labels == 1:
                            loss = loss_fct(logits.squeeze(), labels.squeeze())
                        else:
                            loss = loss_fct(logits, labels)

                    elif self.config.problem_type == "single_label_classification":
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                    elif self.config.problem_type == "multi_label_classification":
                        loss_fct = BCEWithLogitsLoss()
                        loss = loss_fct(logits, labels)

                loss_all.append(loss.item())  # save loss
                if loss_weighted is None:
                    loss_weighted = loss * self.exit_weight_lst[i]
                else:
                    if i < len(self.exit_weight_lst):
                        loss_weighted += loss * self.exit_weight_lst[i]
                    else:  # weight of the last layer
                        loss_weighted += loss * self.ori_weight

            if not return_dict:
                output = (logits,) + outputs[2:]
                result = ((loss_weighted,) + output) if loss is not None else output
            else:
                result = SequenceClassifierOutput(
                    loss=loss_weighted,
                    logits=logits,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
        else:
            # eval mode
            exit_idx = outputs[1]
            self.cnt_exit[exit_idx] += 1
            logits = outputs[2][0]
            if labels is not None:
                if self.config.problem_type == "regression":
                    # We are doing regression
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                else: # multi_label_classification
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)
                if not return_dict:
                    result = (loss, logits) + outputs
                else:
                    result = SequenceClassifierOutput(
                        loss=loss,
                        logits=logits,
                        hidden_states=outputs.hidden_states,
                        attentions=outputs.attentions,
                    )

        return result
