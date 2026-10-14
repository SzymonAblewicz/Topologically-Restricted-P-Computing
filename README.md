# Topologically Restricted P-Computing: Embedding Large Sparse Networks onto Scalable Hardware

Szymon Ablewicz, CRANN, Trinity College Dublin, Ireland. Supervised by Prof. Stefano Sanvito.

The code of my summer 2026 internship report, [`Topologically-Restricted-P-Computing.pdf`](Topologically-Restricted-P-Computing.pdf).

I improved PSSA, the best known heuristic for the King's graph, and extended it with border
constraints (PSSA wBC) so that a large network can be partitioned into small clusters, each on
its own patch with its border nodes on the perimeter, leaving only the cut edges between them
to route. My Divide and Conquer method routes the cut edges by heuristic search, carrying the
classic strategy of electronic design automation onto fixed hardware. My fallback method
instead routes the cut edges through a complete graph embedding of the border nodes alone, so
it is guaranteed, runs in linear time and scales to large problems.

## Figure 23, live

The fallback method on a random cubic graph with 960 vertices at s = 50, replayed from its
recorded run. The whole run plays on [the live page](https://szymonablewicz.github.io/Topologically-Restricted-P-Computing/figures/fig23_simulation/).
The page is `figures/fig23_simulation/`. To open it locally, run `python -m http.server`
in that folder and go to http://localhost:8000.

## Layout

| Folder | Contents | Section |
|---|---|---|
| `src/complete_graph_embeddings/` | the complete graph embedding of Okuyama et al. and TRIAD | 1.3 |
| `src/pssa/ember_on_kings_graph/` | Ember's reconstruction of PSSA, reverted to the King's graph | 2.2 |
| `src/pssa/improved_pssa/` | the improved PSSA | 2.3 |
| `src/pssa/pssa_wbc/` | PSSA with border constraints (PSSA wBC) | 2.4 |
| `src/cluster_embedding/` | the embedding of every cluster, shared by both methods | 3.2 |
| `src/divide_and_conquer/` | the Divide and Conquer method | 3 |
| `src/fallback/` | the fallback method and its two partitioners | 4 |
| `src/benchmark_graphs/` | the benchmark graphs | |
| `experiments/` | the scripts behind the results of the report, by section | |
| `results/` | their data, behind the results of the report | |
| `figures/` | the report's figures, numbered as in the report, and Figure 23 live | |
| `examples/` | small runs of each PSSA, drawn step by step | 2 |
| `tests/` | the fast construction against the full one | 4.1 |
| `familiarization/` | the familiarization stage: p-bits and Camsari's AND gate | |
| `third_party/ember/` | Ember's original PSSA, for D-Wave hardware | |

## Running

Python 3 (tested on 3.14), with

    pip install numpy scipy networkx matplotlib colorcet pybind11
    python build.py

where `build.py` compiles, with a C++17 compiler, the C++ cores of PSSA and of the placing
and routing, and the validator of the large fallback embeddings. Mt-KaHyPar, the fallback's
partitioner, is `pip install mtkahypar` on Linux and runs inside WSL on Windows (see
`src/fallback/mtkahypar_partitioner.py`). Without it, `--method csr` partitions with GGP + T-FM.

    python examples/pssa_wbc.py                                   # PSSA wBC on a 20-vertex graph
    python src/divide_and_conquer/divide_and_conquer.py --graph cubic100
    python src/fallback/run_fallback.py --n 1000 --validate       # a random cubic graph, s = 50

The experiments write their data to `results/`, replacing the saved runs. Every fallback
embedding of Section 4 is a complete and valid minor-embedding, checked directly against the
conditions (M1)-(M3) of the report by `experiments/section4_fallback/validation/`
(`validate_checkpoint.csv` up to 30,000 vertices, `validate_stream.csv` from 30,000 to a
million). The data figures (Figs. 4, 9-12, 17-19 and 22) are redrawn from `results/` by
`figures/fig_section2.py` and `figures/fig_section4.py`, in the report's font when TeX Live
is installed. The embedding drawings are kept as images, and the large outputs of the big
runs (checkpoints, embeddings, partitions) are not included.

## Ember

The PSSA code is built on that of Ember (Macaskill-Smith et al., 2026,
https://github.com/zachmacsmith/ember), under its MIT licence
([`third_party/ember/LICENSE`](third_party/ember/LICENSE)).
