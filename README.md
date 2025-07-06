# MOTIF: Modular Thinking via Reinforcement Fine-tuning in LLMs

🔗 Paper link: [Arxiv preprint](https://arxiv.org/abs/2507.02851)

🔗 Link to the trained models: [Hugging Face collection](https://huggingface.co/collections/purbeshmitra/motif-paper-models-686a2f36407bb88f750eef75)

The [INFTYTHINK architecture](https://arxiv.org/abs/2503.06692v1), shown below, allows multi-round thinking for extended LLM reasoning beyond its context size.
<p align="center">
  <img src="assets/multiround.png" alt="Alt Text" width="750">
</p>

In this work, we propose a GRPO based training method for such a system that allows to calculate the accuracy reward by rolling out trajectories and applying the reward at the first round of inference outcomes. This is depicted as following:
<p align="center">
  <img src="assets/multiround_grpo.png" alt="Alt Text" width="750">
</p>

## Citation
If you find our work useful, consider citing it as:
```bibtex
@article{mitra2025motif,
  title={MOTIF: Modular Thinking via Reinforcement Fine-tuning in LLMs},
  author={Mitra, Purbesh and Ulukus, Sennur},
  journal={arXiv preprint arXiv:2507.02851},
  year={2025}
}
```
