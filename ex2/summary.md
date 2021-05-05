Student 1:
* Name: Guy Bukchin
* ID: 314989252
* Username: guybukchin

Student 2:
* Name: Yuval Alaluf
* ID: 318688710
* Username: yuvalalaluf

Student 3:
* Name: Tomer Ronen
* ID: 308492909
* Username: tomerronen1

Now, for each log file that you need to submit, you will need to write its last 3 lines. For example, this is what we got for `baseline_gen.log`:
```txt
2021-04-24 17:20:46 | INFO | fairseq_cli.generate | NOTE: hypothesis and token scores are output in base 2
2021-04-24 17:20:46 | INFO | fairseq_cli.generate | Translated 7,283 sentences (165,025 tokens) in 18.5s (394.00 sentences/s, 8927.61 tokens/s)
Generate valid with beam=5: BLEU4 = 33.39, 69.1/42.8/28.5/19.4 (BP=0.934, ratio=0.937, syslen=138824, reflen=148229)
```

3 last lines from the baseline_train.log file: 
```txt
2021-04-27 14:56:17 | INFO | fairseq_cli.train | end of epoch 50 (average epoch stats below)
2021-04-27 14:56:17 | INFO | train | epoch 050 | loss 3.943 | nll_loss 2.514 | ppl 5.71 | wps 38426.9 | ups 3.69 | wpb 10419.8 | bsz 422.8 | num_updates 18950 | lr 0.000229718 | gnorm 0.628 | train_wall 50 | gb_free 12.8 | wall 5176
2021-04-27 14:56:17 | INFO | fairseq_cli.train | done training in 5175.6 seconds
```

3 last lines from the baseline_gen.log file: 
```txt
2021-04-27 14:58:39 | INFO | fairseq_cli.generate | NOTE: hypothesis and token scores are output in base 2
2021-04-27 14:58:39 | INFO | fairseq_cli.generate | Translated 7,283 sentences (165,179 tokens) in 26.5s (275.29 sentences/s, 6243.62 tokens/s)
Generate valid with beam=5: BLEU4 = 33.46, 69.1/42.8/28.4/19.4 (BP=0.936, ratio=0.938, syslen=139060, reflen=148229)
```

3 last lines from the baseline_mask.log file: 
```txt
2021-05-05 17:50:40 | INFO | fairseq_cli.generate | NOTE: hypothesis and token scores are output in base 2
2021-05-05 17:50:40 | INFO | fairseq_cli.generate | Translated 7,283 sentences (166,416 tokens) in 27.3s (266.77 sentences/s, 6095.65 tokens/s)
Generate valid with beam=5: BLEU4 = 32.36, 67.8/41.3/27.0/18.1 (BP=0.946, ratio=0.947, syslen=140417, reflen=148229)
```

25 last lines from the check_all_masking_options.log file: 
```txt
2021-05-05 18:59:57 | INFO | fairseq.tasks.translation | /nfs/private/yuval/AMNLP/ex2/data-bin/iwslt14.tokenized.de-en valid de-en 7283 examples
2021-05-05 19:01:19 | INFO | fairseq_cli.generate | NOTE: hypothesis and token scores are output in base 2
2021-05-05 19:01:19 | INFO | fairseq_cli.generate | Translated 7,283 sentences (165,337 tokens) in 25.5s (285.58 sentences/s, 6483.22 tokens/s)
Generate valid with beam=5: BLEU4 = 33.42, 68.9/42.7/28.3/19.3 (BP=0.938, ratio=0.940, syslen=139329, reflen=148229)
table of score with masking enc-enc attention head
rows are transformer layer number and columns are head number
       0      1      2      3
0  33.46  33.42  33.31  33.29
1  33.18  33.41  33.42  33.27
2  33.35  32.43  32.31  33.37
3  33.39  33.13  29.67  33.46
table of score with masking enc-dec attention head
rows are transformer layer number and columns are head number
       0      1      2      3
0  33.32  33.23  16.26  33.23
1  33.19  33.43  33.01  33.25
2  33.34  33.49  32.82  32.89
3  32.36  32.86  32.42  32.88
table of score with masking dec-dec attention head
rows are transformer layer number and columns are head number
       0      1      2      3
0  33.41  33.40  33.46  33.38
1  33.46  33.40  33.32  33.40
2  33.32  33.38  33.02  33.35
3  33.39  33.45  33.33  33.42
```

3 last lines from the sandwich_train.log file: 
```txt
<write here>
```

3 last lines from the sandwich_gen.log file: 
```txt
<write here>
```