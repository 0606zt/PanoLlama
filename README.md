# 🦙PanoLlama: Generating Endless and Coherent Panoramas with Next-Token-Prediction LLMs

![intro](docs/intro.png)

## Introduction

What is in **PanoLlama**:

> 1. **New Paradigm**: We define a new paradigm for PIG, modeling it as a **next-token prediction** task to better solve
     the multilevel coherence challenge.
> 2. **New Strategy**: Based on token redirection, we develop a training-free **next-crop prediction** strategy that
     enables endless PIG with existing VAR models. Compared to current methods with complex designs, PanoLlama offers a
     more straightforward and efficient framework, achieving SOTA performance in coherence (47.50%), fidelity \&
     diversity (28.16%), and aesthetics (15%).
> 3. **Additional Applications**: Beyond basic panorama generation, we support applications other PIG methods cannot
     achieve, including multi-scale generation, mask-free layout control, and multi-guidance synthesis.
> 4. **New Benchmark**: Given the lack of a standardized testing prompt in prior PIG works, which typically rely on 5-20
     specific ones, we construct a dataset of 1,000 detailed prompts across 100+ themes. Along with a comprehensive set
     of baselines and metrics, this establishes a new benchmark for panorama generation.

For more details, please visit our [paper page](https://arxiv.org/abs/2411.15867).

## Get Started

**Configuration** &ensp; Set up and configure the environment by installing the required packages:

```bash
pip install -r requirements.txt
```

**Pre-trained Models** &ensp; Download pre-trained models $\Phi$
from [LlamaGen](https://github.com/FoundationVision/LlamaGen), and place them in the folder `/models` under the
corresponding modules:

|     module      |    model    | params | tokens |                                                 weight                                                 |
|:---------------:|:-----------:|:------:|:------:|:------------------------------------------------------------------------------------------------------:|
|  text encoder   | FLAN-T5-XL  |   3B   |   /    |                    [flan-t5-xl](https://huggingface.co/google/flan-t5-xl/tree/main)                    |
| image tokenizer |    VQVAE    |  72M   | 16x16  |       [vq_ds16_t2i.pt](https://huggingface.co/peizesun/llamagen_t2i/resolve/main/vq_ds16_t2i.pt)       |
| token generator | LlamaGen-XL |  775M  | 32x32  | [t2i_XL_stage2_512.pt](https://huggingface.co/peizesun/llamagen_t2i/resolve/main/t2i_XL_stage2_512.pt) |

**Generation** &ensp; We support panorama expansion in vertical, horizontal, and both directions. Try the following
command to generate a horizontal one:

```bash
python -m token_generator.sample \
    --seed -1 \
    --times 12 \
    --addit-cols 24 \
    --lam 1 \
    --gen-mode h \
    --n 1
```

## Citation

If you find our work helpful, please consider citing:

```bibtex
@article{zhou2024panollama,
  title={PanoLlama: Generating Endless and Coherent Panoramas with Next-Token-Prediction LLMs},
  author={Zhou, Teng and Zhang, Xiaoyu and Tang, Yongchuan},
  journal={arXiv preprint arXiv:2411.15867},
  year={2024}
}
```
