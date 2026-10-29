# Emergence of functional directionality in structurally asymmetric neuronal cultures

Data and code of the article *Emergence of Functional Directionality in Structurally
Asymmetric Neuronal Cultures: an in Vitro and in Silico Approach*.

Each numbered folder is one step of the analysis; run them in order (see "How to run").
Every plot and number of the article is produced here, except drawings and microscopy images
(Figs. 1, 2a, 3c, 4) and the network images of Figs. 6a–b, made with Netcal and Gephi from
`software_outputs/GTE/15_05_1/DIV7`.

```
├── common/                   Code shared by all steps
│   ├── bursts.py             Chamber GNA, burst detection and matching between chambers
│   ├── recordings.py         Folder locations, recording names and culture labels (C1–C8)
│   └── plot_style.py         Text sizes and colours of all figures
│
├── 1_raw_data/               All baseline recordings (50 recordings, 8 cultures)
├── software_outputs/         Output of the MATLAB software of the laboratory (code not included)
│   ├── GTE/                  Effective connectivity networks (z >= 2) and their metrics
│   └── VisualizeFront/       GNA of the whole network and parameters used
│
├── 2_screening/              Valid recordings                          -> recordings.csv
├── 3_window/                 Coincidence window and chamber threshold  -> Fig. SF3
├── 4_culture_selection/      Coupled cultures                          -> Table ST1 (cultures.csv)
├── 5_analysis_data/          Recordings of the 7 coupled cultures (36), created by step 4
├── 6_directionality/         Direction of propagation                  -> Figs. 5e, 5f, 5g
├── 7_example_culture/        Example culture C1, DIV 17                -> Figs. 5a–d
├── 8_topology/               Topology of the GTE networks              -> Fig. 6c, Table ST2 (topology_cross_chamber.csv)
├── 9_raster_gna_example/     Raster and GNA of C1, DIV 21              -> Fig. 2b
└── 10_in_silico/             Computational model (notebook)            -> Figs. 3a, 3b, 3d, 3e, 5h, 5i, 5j (figures/)
```

## Data

- `spikesData.csv`: ROI and time (s) of every spike, exported from Netcal.
  `ROI_centers_everything.csv`: position of every ROI.
- Chamber A = ROIs 1–200, chamber B = ROIs 201–400. The source is the chamber at the wide end of
  the funnel-shaped microchannels (given for each recording in `5_analysis_data/index.csv`).
- Cultures are named by date. In the article: C1 = 15_05_1, C2 = 15_05_2, C3 = 30_04_2,
  C4 = 20_02, C5 = 30_04_3, C6 = 22_05, C7 = 30_04_1, C8 = 20_04 (not coupled, excluded).

## The steps

- **2 – Screening.** Excludes recordings after DIV 24, recordings with no bursts in one chamber,
  and two faulty recordings (20_04 DIV7: wrong frame rate; 30_04_1 DIV11: abnormal activity).
- **Bursts in each chamber** (`common/bursts.py`). Chamber GNA with a 0.5 s window, threshold 0.1;
  each burst is represented by the time of its peak.
- **3 – Window.** Compares the time between bursts of the two chambers with chance (circular
  time shifts) and sets delta t = 1 s. `threshold_check.py` compares the thresholds 0.1 and 0.2.
- **4 – Culture selection.** Keeps the cultures with more coincident bursts than chance (p < 0.05).
- **6 – Directionality.** Bursts of the two chambers peaking within delta t = 1 s are paired; the
  first chamber is the origin (S->T or T->S), unpaired bursts are local. Counts are summed per
  culture (the statistical unit) to compute the directionality index, the recruitment score
  (DIV 14–24) and a binomial test per culture.
- **7 – Example culture.** Same analysis for one recording; initiation site = centroid of the
  first 20% of ROIs to fire in the origin chamber.
- **8 – Topology.** Reads `software_outputs/GTE`. GTE was computed on the whole recording (33 ms
  bins, Markov order 2, instant feedback) after removing repeated spikes of the same ROI within
  1 s; connections with z >= 2 were kept. Networks with < 100 active nodes or < 80% of them
  connected are excluded.
- **9 – Raster and GNA.** Spikes from `1_raw_data`, whole-network GNA (1 s window, threshold 0.2)
  from `software_outputs/VisualizeFront`.
- **10 – In silico model.** All parameters are described in the notebook. `in_silico_networks.csv`
  contains the values of each of the 10 simulated networks.

## How to run

```
python3 -m pip install -r requirements.txt

cd 2_screening             && python3 screening.py
cd ../3_window             && python3 window.py && python3 threshold_check.py
cd ../4_culture_selection  && python3 selection.py
cd ../6_directionality     && python3 directionality.py
cd ../7_example_culture    && python3 example_culture.py
cd ../8_topology           && python3 topology.py
cd ../9_raster_gna_example && python3 raster_gna.py
cd ../10_in_silico         && jupyter nbconvert --to notebook --execute --inplace in_silico.ipynb   # ~10 min
```

Figures are saved as PNG (300 dpi) and PDF. Text sizes and colours: `common/plot_style.py`.
