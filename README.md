# AMPRecognitionBenchmark

The official code for the paper "A Benchmark for Antimicrobial Peptide Recognition Based on Structure and Sequence Representation"

Antimicrobial peptides (AMPs) serve as potent therapeutic agents against drug-resistant (DR) microbes; however, their clinical application is constrained by limitations in activity. Recently, machine learning has shown significant promise in recognizing high-activity AMPs. Nevertheless, these activity datasets about AMPs are aggregated from thousands of publications, which employ varying wet-lab experimental setups and focus on only one or a few types of DR bacteria. This heterogeneity restricts the advancement of AI methods for fair evaluation of AMP recognition. Additionally, while AlphaFold has revolutionized drug discovery through accurate protein structure predictions, the integration of these predicted structures into AMP discovery remains unexplored.

To address these challenges, we present two key contributions:

**(a) DRAMPAtlas 1.0**: We introduce **DRAMPAtlas 1.0**, comprising a training set sourced from public databases and a testing set derived from our wet-lab experiments. Each AMP sequence in the atlas is annotated with its 3D structure, activity data against six types of DR bacteria, and toxicity profiles.

**(b) Comprehensive AMP recognition Experiments**: We perform extensive experiments on AMP recognition by modeling the 3D structures as voxels or graphs, either in combination with sequence information or using structure or sequence data exclusively. Our experiments reveal several insightful findings that enhance our understanding of AMP activity prediction.

We anticipate that our benchmark and findings will aid the research community in designing more effective algorithms for discovering high-activity AMPs. 

![main](https://github.com/EricwanAR/AMPRecognitionBenchmark/blob/main/pics/benchmark.png)

## DRAMPAtlas 1.0 Dataset
The full atlas is on the [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FE9A88D). The wet-lab test set is not in that deposit yet and will be released after publication.

This repository includes only a small sample, so the loaders and released checkpoints can be inspected without the full atlas:

- `metadata/data_0920_i.csv`: 8 training peptides
- `metadata/data_qlx.csv`: 4 wet-lab test peptides
- `metadata/data_saap.csv`: 4 SAAP peptides
- `metadata/data_simi.csv`: 4 sequences from the similarity-exclusion list
- `metadata/features.txt`: amino-acid features used by the graph models
- `pdb/pdb_af/` and `pdb/pdb_dbassp/`: the AlphaFold and HelixFold structure for each sample peptide

These rows are examples. They are not the training split used in the paper.

### Full dataset
1. Put the downloaded `dataverse_files.zip` in `AMPRecognitionBenchmark/`.
2. Run `bash dataproc.sh`.
3. This replaces the sample `metadata/` and `pdb/` folders with the full tables and structures:

```bash
metadata/
├── data_simi.csv
├── data_0920_i.csv
└── ...

pdb/
├── pdb_af/ #AlphaFold predictions
│   └── (pdb files ...)
└── pdb_dbassp/ #HelixFold predictions
    └── (pdb files ...)
```

Graph models also read `metadata/features.txt` from the unit directory (`t3.2.2`, `t5g`, `t7g`, `graphabl`). That file is already included.

## Instruction
### Download
```bash
git clone https://github.com/EricwanAR/AMPRecognitionBenchmark.git
cd AMPRecognitionBenchmark
```

### Training and inference
1. `cd` to any unit folder, for example `cd t3.1` or `cd voxabl`.
2. `python main.py [args]`. Arguments are in `main.py`. `runthro.sh` lists the settings used in the paper.
3. Checkpoints are written to `run/`. `model_k.pth` is the epoch with the best validation score for that fold. The `qlx` block in `result.txt` is selected separately and is not the score of that checkpoint.
4. Released checkpoints are already in each unit's `run/` directory, together with `config.json` and `result.txt`.
5. For inference, use `infer.py` with the same arguments, for example `python infer.py [training args]`. The sample metadata and PDB files are enough to check that a checkpoint loads. Reported numbers use the full atlas.