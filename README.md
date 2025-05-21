# AMPRecognitionBenchmark

The official code for the paper "A Benchmark for Antimicrobial Peptide Recognition Based on Structure and Sequence Representation"

Antimicrobial peptides (AMPs) serve as potent therapeutic agents against drug-resistant (DR) microbes; however, their clinical application is constrained by limitations in activity. Recently, machine learning has shown significant promise in recognizing high-activity AMPs. Nevertheless, these activity datasets about AMPs are aggregated from thousands of publications, which employ varying wet-lab experimental setups and focus on only one or a few types of DR bacteria. This heterogeneity restricts the advancement of AI methods for fair evaluation of AMP recognition. Additionally, while AlphaFold has revolutionized drug discovery through accurate protein structure predictions, the integration of these predicted structures into AMP discovery remains unexplored.

To address these challenges, we present two key contributions:

**(a) DRAMPAtlas 1.0**: We introduce **DRAMPAtlas 1.0**, comprising a training set sourced from public databases and a testing set derived from our wet-lab experiments. Each AMP sequence in the atlas is annotated with its 3D structure, activity data against six types of DR bacteria, and toxicity profiles.

**(b) Comprehensive AMP recognition Experiments**: We perform extensive experiments on AMP recognition by modeling the 3D structures as voxels or graphs, either in combination with sequence information or using structure or sequence data exclusively. Our experiments reveal several insightful findings that enhance our understanding of AMP activity prediction.

We anticipate that our benchmark and findings will aid the research community in designing more effective algorithms for discovering high-activity AMPs. 

![main](https://github.com/EricwanAR/AMPRecognitionBenchmark/blob/main/pics/benchmark.png)

## DRAMPAtlas 1.0 Dataset
[Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FE9A88D)(wet-lab data not included for now, reveal after publication)
