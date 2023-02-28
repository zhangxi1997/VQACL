from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from modeling_t5_our import VLT5

class VLT5VQA(VLT5):
    def __init__(self, config, num_answers=None, label2ans=None):
        super().__init__(config) # 初始化VLT5

        self.num_answers = num_answers
        self.label2ans = label2ans
        self.bce_loss = nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def test_step(self, batch, **kwargs):
        self.eval()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)


        result = {}
        if self.config.classifier:
            B = len(input_ids)

            decoder_input_ids = torch.ones(
                B, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id

            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                cate_labels=cate_labels,
                ques_labels=ques_labels,
                proto_update=False,
                return_dict=True
            )

            last_layer_hidden_state = output.decoder_hidden_states[-1]
            last_hidden_state = last_layer_hidden_state.view(B, -1, self.config.d_model)[:, -1]

            # [B, num_answers]
            logit = self.answer_head(last_hidden_state)

            score, pred_ans_id = logit.max(1)
            pred_ans_id = pred_ans_id.cpu().numpy()
            pred_ans = [self.label2ans[ans_id] for ans_id in pred_ans_id]

            result['pred_ans'] = pred_ans

        else:

            output = self.generate(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                **kwargs,
            )
            generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            result['token_ids'] = output
            result['pred_ans'] = generated_sents

        return result
