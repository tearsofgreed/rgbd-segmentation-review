from typing import Dict, List, Any

from torch import Tensor


def mask2former_forward_with_intermediate(head, x: List[Tensor]) -> Dict[str, Any]:
    """外部 helper：显式提取 Mask2FormerHead 中间状态。

    注意：
    1. 这是一个外部函数，不挂在训练用 head 类里
    2. 只在你明确需要中间特征时调用
    3. 默认不参与普通 baseline 训练路径
    """
    batch_size = x[0].shape[0]

    mask_features, multi_scale_memorys = head.pixel_decoder(x)

    decoder_inputs = []
    decoder_positional_encodings = []

    for i in range(head.num_transformer_feat_level):
        decoder_input = head.decoder_input_projs[i](multi_scale_memorys[i])
        decoder_input = decoder_input.flatten(2).permute(0, 2, 1)

        level_embed = head.level_embed.weight[i].view(1, 1, -1)
        decoder_input = decoder_input + level_embed

        mask = decoder_input.new_zeros(
            (batch_size,) + multi_scale_memorys[i].shape[-2:],
            dtype=bool
        )
        decoder_positional_encoding = head.decoder_positional_encoding(mask)
        decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(0, 2, 1)

        decoder_inputs.append(decoder_input)
        decoder_positional_encodings.append(decoder_positional_encoding)

    query_feat = head.query_feat.weight.unsqueeze(0).repeat((batch_size, 1, 1))
    query_embed = head.query_embed.weight.unsqueeze(0).repeat((batch_size, 1, 1))

    cls_pred_list = []
    mask_pred_list = []
    query_state_list = [query_feat]

    cls_pred, mask_pred, attn_mask = head._forward_head(
        query_feat,
        mask_features,
        multi_scale_memorys[0].shape[-2:]
    )
    cls_pred_list.append(cls_pred)
    mask_pred_list.append(mask_pred)

    for i in range(head.num_transformer_decoder_layers):
        level_idx = i % head.num_transformer_feat_level

        mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
        attn_mask = attn_mask & mask_sum

        layer = head.transformer_decoder.layers[i]
        query_feat = layer(
            query=query_feat,
            key=decoder_inputs[level_idx],
            value=decoder_inputs[level_idx],
            query_pos=query_embed,
            key_pos=decoder_positional_encodings[level_idx],
            cross_attn_mask=attn_mask,
            query_key_padding_mask=None,
            key_padding_mask=None
        )

        query_state_list.append(query_feat)

        cls_pred, mask_pred, attn_mask = head._forward_head(
            query_feat,
            mask_features,
            multi_scale_memorys[(i + 1) % head.num_transformer_feat_level].shape[-2:]
        )
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

    return dict(
        cls_pred_list=cls_pred_list,
        mask_pred_list=mask_pred_list,
        query_state_list=query_state_list,
        query_feat_last=query_feat,
        coarse_mask_pred=mask_pred_list[-1],
        mask_features=mask_features,
        decoder_inputs=decoder_inputs,
        decoder_positional_encodings=decoder_positional_encodings,
        query_embed=query_embed,
        multi_scale_memorys=multi_scale_memorys
    )
