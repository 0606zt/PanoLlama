# Modified from:
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
import torch
import torch.nn as nn
import torchvision.transforms.functional
from torch.nn import functional as F
import torch._dynamo.config
import torch._inductor.config
import copy
from einops import rearrange


# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.triton.unique_kernel_names = True
# torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future


# from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
        logits,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample(
        logits,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        sample_logits=True
):
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs


def logits_to_probs(
        logits,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = None,
        **kwargs
):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def prefill(
        model,
        cond_idx: torch.Tensor,
        input_pos: torch.Tensor,
        cfg_scale: float,
        **sampling_kwargs
):
    if cfg_scale > 1.0:
        logits, _ = model(None, cond_idx, input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits, _ = model(None, cond_idx, input_pos)
    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(
        model,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        cfg_scale: float,
        cfg_flag: bool,
        **sampling_kwargs
):
    assert input_pos.shape[-1] == 1
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits, _ = model(x_combined, cond_idx=None, input_pos=input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        if cfg_flag:
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = cond_logits
    else:
        logits, _ = model(x, cond_idx=None, input_pos=input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(
        model,
        cur_token: torch.Tensor,
        input_pos: torch.Tensor,
        num_tokens: int,
        cfg_scale: float,
        cfg_interval: int,
        **sampling_kwargs
):
    new_tokens, new_probs = [], []
    cfg_flag = True
    for i in range(num_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):  # actually better for Inductor to codegen attention here
            if -1 < cfg_interval < i:
                cfg_flag = False
            next_token, next_prob = decode_one_token(model, cur_token, input_pos, cfg_scale, cfg_flag, **sampling_kwargs)
            input_pos += 1
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(-1, 1)
    return new_tokens, new_probs


@torch.no_grad()
def generate(
        model,
        cond: torch.Tensor, num_tokens, emb_masks: torch.Tensor = None,
        cfg_scale=1.0, cfg_interval=-1,
        gen_mode: str = 'h', times=3, addit_rows=16, addit_cols=16,
        lam=1.0,
        **sampling_kwargs
):
    if model.model_type != 't2i':
        raise Exception("Please check model type!")

    batch_size = cond.shape[0]
    device = cond.device
    latent_size = int(num_tokens ** 0.5)

    # generate first token block
    if cfg_scale > 1.0:
        cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
        cond_combined = torch.cat([cond, cond_null])
    else:
        cond_combined = cond
    T = cond.shape[1]

    with torch.device(device):
        batch_size_cfg = batch_size * 2 if cfg_scale > 1.0 else batch_size
        model.setup_caches(max_batch_size=batch_size_cfg, max_seq_length=T + num_tokens, dtype=model.tok_embeddings.weight.dtype)

    if emb_masks is not None:
        assert emb_masks.shape[0] == batch_size
        assert emb_masks.shape[-1] == T
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
        else:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)
        eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix

    seq = torch.empty((batch_size, T + num_tokens), dtype=torch.int, device=device)  # create an empty tensor of the expected final shape and fill in the current tokens

    input_pos = torch.arange(0, T, device=device)
    cur_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)
    seq[:, T:T + 1] = cur_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens, _ = decode_n_tokens(model, cur_token, input_pos, num_tokens - 1, cfg_scale, cfg_interval, **sampling_kwargs)
    seq[:, T + 1:] = torch.cat(generated_tokens, dim=1)

    # generate additional tokens, supporting vertical, horizontal, and both directions
    seqs = [seq[:, T:]]
    if gen_mode == 'v':
        pano_seq = seqs[-1]
        addit_num_tokens = addit_rows * latent_size
        for t in range(times):
            addit_seq = generate_additional_tokens_vertical(model, num_tokens, cfg_scale, cfg_interval, sampling_kwargs,
                                                            T, device, generated_tokens=seqs[-1],
                                                            addit_num_tokens=addit_num_tokens)
            seqs.append(torch.cat([seqs[-1][:, addit_num_tokens:], addit_seq], dim=1))
            pano_seq = torch.cat([pano_seq, addit_seq], dim=1)
    elif gen_mode == 'h':
        pano_seq = rearrange(seqs[-1], "b (h w) -> b h w", h=latent_size)
        for t in range(times):
            addit_seq = generate_additional_tokens_horizontal(model, cfg_scale, sampling_kwargs, T, latent_size, device,
                                                              generated_tokens=seqs[-1], addit_cols=addit_cols)
            seqs.append(addit_seq)
            addit_seq = rearrange(addit_seq, "b (h w) -> b h w", h=latent_size)
            pano_seq = torch.cat([pano_seq, addit_seq[:, :, latent_size - addit_cols:]], dim=2)
    else:
        times_v = times_h = times
        pano_seq = []
        addit_num_tokens = addit_rows * latent_size
        for t_v in range(times_v + 1):
            pano_seq_h = rearrange(seqs[-1], "b (h w) -> b h w", h=latent_size)
            for t_h in range(times_h):
                addit_seq = generate_additional_tokens_horizontal(model, cfg_scale, sampling_kwargs, T, latent_size, device,
                                                                  generated_tokens=seqs[-1], addit_cols=addit_cols)
                seqs.append(addit_seq)
                addit_seq = rearrange(addit_seq, "b (h w) -> b h w", h=latent_size)
                pano_seq_h = torch.cat([pano_seq_h, addit_seq[:, :, latent_size - addit_cols:]], dim=2)
            pano_seq_h = pano_seq_h if t_v == 0 else pano_seq_h[:, latent_size - addit_rows:, :]
            pano_seq.append(pano_seq_h)
            addit_seq = generate_additional_tokens_vertical(model, num_tokens, cfg_scale, cfg_interval, sampling_kwargs,
                                                            T, device, generated_tokens=seqs[t_v * (times_h + 1)],
                                                            addit_num_tokens=addit_num_tokens)
            seqs.append(torch.cat([seqs[-1][:, addit_num_tokens:], addit_seq], dim=1))
        pano_seq = torch.cat(pano_seq, dim=1)
    return pano_seq, seqs


def generate_additional_tokens_vertical(
        model,
        num_tokens,
        cfg_scale, cfg_interval, sampling_kwargs, T, device,
        generated_tokens: torch.Tensor, addit_num_tokens: int
):
    cur_token = generated_tokens[:, -1].unsqueeze(1)  # the last one of generated tokens
    input_pos = torch.tensor([T + num_tokens - addit_num_tokens], device=device, dtype=torch.int)
    addit_tokens, _ = decode_n_tokens(model, cur_token, input_pos, addit_num_tokens, cfg_scale, cfg_interval, **sampling_kwargs)
    addit_seq = torch.cat(addit_tokens, dim=1)
    return addit_seq


def generate_additional_tokens_horizontal(
        model,
        cfg_scale, sampling_kwargs, T, latent_size, device,
        generated_tokens: torch.Tensor, addit_cols: int
):
    generated_tokens = rearrange(generated_tokens, "b (h w) -> b h w", h=latent_size)
    updated_tokens = torch.randint(0, 1, generated_tokens.shape, device=device) - 1
    updated_tokens[:, :, :latent_size - addit_cols] = generated_tokens[:, :, addit_cols:]
    updated_tokens = rearrange(updated_tokens, "b h w -> b (h w)", h=latent_size)
    cur_token = updated_tokens[:, 31 - addit_cols].unsqueeze(1)  # the last token of each row of generated tokens
    for i in range(latent_size - addit_cols, 1024):
        input_pos = torch.tensor([T + i - 1], device=device, dtype=torch.int)
        next_token, _ = decode_one_token(model, cur_token, input_pos, cfg_scale, True, **sampling_kwargs)
        if torch.all(updated_tokens[:, i] == -1):
            updated_tokens[:, i] = next_token.squeeze(1)
        # else:
        #     updated_tokens[:, i] = average_tokens(vq_model, updated_tokens[:, i], next_token.squeeze(1), lam=0.8)  # lam=1.0 achieves the best performance, equivalent to commenting out this line
        cur_token = updated_tokens[:, i].unsqueeze(1)
    return updated_tokens


def average_tokens(vq_model, token1, token2, lam: float):
    if vq_model.l2_norm:
        embedding = F.normalize(vq_model.embedding.weight, p=2, dim=-1)
    embedding1, embedding2 = embedding[token1], embedding[token2]
    ave_embedding = lam * embedding1 + (1 - lam) * embedding2
    d = torch.sum(ave_embedding ** 2, dim=1, keepdim=True) + \
        torch.sum(embedding ** 2, dim=1) - 2 * \
        torch.einsum('bd,dn->bn', ave_embedding, torch.einsum('n d -> d n', embedding))
    ave_tokens = torch.argmin(d, dim=1)
    return ave_tokens


def rotater(vq_model, qzshape, seq):
    samples = vq_model.decode_code(seq, qzshape)
    rotated_samples = torch.rot90(samples, -1, [2, 3])  # rotate image samples 90° clockwise
    rotated_tokens = vq_model.encode_code(rotated_samples)
    rotated_tokens = rearrange(rotated_tokens, "b h w-> b (h w)")
    return rotated_tokens
