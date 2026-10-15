# X-CLIP Mini for RTX 4060

This branch supports truncating the pretrained CLIP text and visual encoders.
The default 12+12-layer architecture remains unchanged unless the new layer
arguments are supplied.

## Recommended 8 GB profile

The included MSR-VTT recipe is designed as a conservative starting point for a
single RTX 4060 with 8 GB VRAM:

- CLIP ViT-B/32 with 6 visual and 6 text Transformer blocks
- 6 uniformly sampled video frames
- 24 text tokens
- parameter-free `meanP` similarity header
- 4 samples per optimizer micro-step
- gradient accumulation of 4, for an effective batch size of 16
- the first 4 retained CLIP blocks frozen

Set `DATA_PATH` in the script and run it under Linux or WSL2:

```bash
sh scripts/run_xclip_mini_msrvtt_rtx4060.sh
```

The important architecture options are:

```text
--visual_num_hidden_layers 6
--text_num_hidden_layers 6
--max_frames 6
--max_words 24
--loose_type
--sim_header meanP
```

Both layer counts must be between 1 and the number of layers in the selected
pretrained CLIP checkpoint. The retained blocks reuse their pretrained weights;
surplus blocks are ignored while loading. Training logs report both total and
trainable parameter counts after freezing.

If CUDA runs out of memory, reduce `--batch_size` from 16 to 8 while keeping
`--gradient_accumulation_steps=4`. This changes the micro-batch from 4 to 2.
