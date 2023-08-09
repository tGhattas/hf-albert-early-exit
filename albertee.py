from transformers import AlbertForSequenceClassification, AlbertConfig, AlbertModel
from transformers.models.albert.modeling_albert import AlbertTransformer, AlbertModel, BaseModelOutput, AlbertLayerGroup,\
    SequenceClassifierOutput, AlbertPreTrainedModel, CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch import nn
from typing import Dict, List, Optional, Tuple, Union
import torch


# class AlbertTransformerEarlyExit(AlbertTransformer):
#     def __init__(self, config: AlbertConfig):
#         super().__init__(config)
#
#         self.config = config
#         self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
#         self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])
#         self.exit_layer = nn.Linear(config.hidden_size, 1)
#
#     def forward(
#             self,
#             hidden_states: torch.Tensor,
#             attention_mask: Optional[torch.FloatTensor] = None,
#             head_mask: Optional[torch.FloatTensor] = None,
#             output_attentions: bool = False,
#             output_hidden_states: bool = False,
#             return_dict: bool = True,
#     ) -> Union[BaseModelOutput, Tuple]:
#         hidden_states = self.embedding_hidden_mapping_in(hidden_states)
#
#         all_hidden_states = (hidden_states,) if output_hidden_states else None
#         all_attentions = () if output_attentions else None
#
#         head_mask = [None] * self.config.num_hidden_layers if head_mask is None else head_mask
#
#         for i in range(self.config.num_hidden_layers):
#             # Number of layers in a hidden group
#             layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)
#
#             # Index of the hidden group
#             group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))
#
#             layer_group_output = self.albert_layer_groups[group_idx](
#                 hidden_states,
#                 attention_mask,
#                 head_mask[group_idx * layers_per_group: (group_idx + 1) * layers_per_group],
#                 output_attentions,
#                 output_hidden_states,
#             )
#             hidden_states = layer_group_output[0]
#             # exit in case exit gate passes confidence threshoold
#             # if self.exit_layer(hidden_states) > 0.5:
#             #     break
#             print(self.exit_layer(hidden_states).shape, '-'*100)
#
#             if output_attentions:
#                 all_attentions = all_attentions + layer_group_output[-1]
#
#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)
#
#         if not return_dict:
#             return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
#         return BaseModelOutput(
#             last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
#         )


# class AlbertModelEarlyExit(AlbertModel):
#     def __init__(self, config: AlbertConfig, add_pooling_layer: bool = True):
#         super().__init__(config, add_pooling_layer)
        # self.encoder = AlbertTransformer(config)


class AlbertForSequenceClassificationEarlyExit(AlbertPreTrainedModel):
    def __init__(self, config: AlbertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

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
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

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

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

