import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from torchvision.utils import save_image

import os
import argparse
from image_tokenizer.vq_model import VQ_models
from text_encoder.t5 import T5Embedder
from gpt import GPT_models
from generate import generate

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# a naive way to implement left-padding
def process_left_padding(caption_embs, emb_masks):
    new_emb_masks = torch.flip(emb_masks, dims=[-1])
    new_caption_embs = []
    for i, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
        valid_num = int(emb_mask.sum().item())
        print(f"  prompt {i} token len: {valid_num}")
        new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
        new_caption_embs.append(new_caption_emb)
    new_caption_embs = torch.stack(new_caption_embs)
    return new_caption_embs, new_emb_masks


def main(args):
    # setup pytorch
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_grad_enabled(False)
    device = 'cuda:1'

    # load vq model
    print("Loading vq model...")
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint['model'])
    del checkpoint

    # load gpt model
    print("Loading gpt model...")
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        block_size=latent_size ** 2,
        cls_token_num=args.cls_token_num,
        model_type='t2i',
    ).to(device=device, dtype=precision)
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if "model" in checkpoint:  # ddp
        model_weight = checkpoint['model']
    elif "module" in checkpoint:  # deepspeed
        model_weight = checkpoint['module']
    elif "state_dict" in checkpoint:
        model_weight = checkpoint['state_dict']
    else:
        raise Exception("please check model weight")
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint

    # compile gpt model if needed
    if args.compile:
        print("Compiling gpt model...")
        # requires PyTorch 2.0 (optional)
        gpt_model = torch.compile(
            gpt_model,
            mode='reduce-overhead',
            fullgraph=True
        )

    # load t5 model
    print("Loading t5 model...")
    t5_model = T5Embedder(
        device=device,
        local_cache=True,
        cache_dir=args.t5_path,
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )

    # prompts -> conditional embeddings and masks
    caption_embs, emb_masks = t5_model.get_text_embeddings(args.prompts)
    if not args.no_left_padding:
        print("Processing left-padding for prompts...")
        caption_embs, emb_masks = process_left_padding(caption_embs, emb_masks)
    cond_tokens = caption_embs * emb_masks[:, :, None]

    # generate tokens
    print("Generating tokens...")
    pano_tokens, block_tokens = generate(
        model=gpt_model,
        cond=cond_tokens, num_tokens=latent_size ** 2, emb_masks=emb_masks,
        cfg_scale=args.cfg_scale,
        gen_mode=args.gen_mode, times=args.times, addit_rows=args.addit_rows, addit_cols=args.addit_cols,
        temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, sample_logits=True,
    )

    # tokens -> panoramas
    print("Decoding tokens into images...")
    addit_latent_size_col = 0 if args.gen_mode == 'v' else args.times * args.addit_cols
    addit_latent_size_row = 0 if args.gen_mode == 'h' else args.times * args.addit_rows
    pano_qzshape = [len(cond_tokens), args.codebook_embed_dim, latent_size + addit_latent_size_row, latent_size + addit_latent_size_col]  # [n,8,32+addit_row,32+addit_col]
    pano_samples = vq_model.decode_code(pano_tokens, pano_qzshape)

    # save panoramas
    save_image(pano_samples, f"../output.png", nrow=1, normalize=True, value_range=(-1, 1), padding=0)

    # if record blocks
    qzshape = [len(cond_tokens), args.codebook_embed_dim, latent_size, latent_size]  # [n,8,32,32]
    for i, block_tokens_i in enumerate(block_tokens):
        block_samples = vq_model.decode_code(block_tokens_i, qzshape)
        save_image(block_samples, f"../output_block{i}.png", nrow=4, normalize=True, value_range=(-1, 1), padding=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5-path", type=str, default="../text_encoder/models")
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--no-left-padding", action='store_true', default=False)
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default="./models/t2i_XL_stage2_512.pt")
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default="../image_tokenizer/models/vq_ds16_t2i.pt", help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--top-k", type=int, default=1000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--prompts", type=str, default="Generate a sprawling, verdant countryside landscape with rolling hills, winding streams, and distant mountains. The scene should have a classic, pastoral aesthetic, with lush green meadows dotted with vibrant wildflowers and clusters of deciduous trees. A small, rustic village with stone buildings and thatched roofs should sit nestled in a valley, surrounded by neatly tended farmland and orchards. A dirt road should wind its way through the landscape, flanked by picket fences and the occasional weathered barn or silo. The sky should be a warm, hazy blue, with fluffy white clouds drifting lazily overhead. The overall mood should be one of tranquility, timelessness, and the natural beauty of the countryside. The color palette should be muted and earthy, with plenty of greens, browns, grays, and ochres. The lighting should be soft and diffused, creating long shadows and highlights that give depth and dimensionality to the scene. Attention to detail is important, so include small touches like wildflowers, grazing livestock, and the smoke rising from chimneys in the village. This landscape should have a distinctly European, almost storybook-like quality, evoking the pastoral landscapes of Normandy, the English countryside, or the rolling hills of Tuscany. The entire scene should feel cohesive and harmonious, with a sense of balance and proportion that makes it feel like a complete, self-contained world.")
    parser.add_argument("--times", type=int, default=12, help="the number of times additional tokens are generated")
    parser.add_argument("--addit-rows", type=int, default=16, help="additional token rows generated each time")  # limit:[0,31]
    parser.add_argument("--addit-cols", type=int, default=24, help="additional token columns generated each time")  # limit:[0,31], recommend:[16,30]
    parser.add_argument("--lam", type=float, default=1.0, help="average factor of token embeddings")
    parser.add_argument('--gen-mode', type=str, choices=['h', 'v', 'both'], default='h', help="the orientation of generation, horizontal, vertical, or both")
    parser.add_argument('--n', type=int, default=1, help="number of panoramas to generate")
    args = parser.parse_args()
    args.prompts = [args.prompts] * args.n
    main(args)

