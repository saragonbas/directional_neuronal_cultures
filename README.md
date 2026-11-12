# Dynamic and complex network analysis of asymmetric neuronal cultures

Data and code of the article *Dynamic and Complex Network Analysis of Asymmetric Neuronal
Cultures: an in Vitro and in Silico Approach*.

Every plot and number of the article comes from the code in this folder, except the drawings
and microscopy images (Figs. 1, 2a, 3c and 4) and the network images of Figs. 6a–b, which were
made with Netcal and Gephi from `software_outputs/GTE/15_05_1/DIV7`.

Each numbered folder is one step of the analysis and they are run in order (see *How to run*).

```
├── common/                   Code shared by all the steps
│   ├── bursts.py             Bursts of each chamber and pairing of bursts between chambers
│   ├── recordings.py         Folder locations, recording names and culture labels (C1–C8)
│   └── plot_style.py         Text sizes and colours of the figures
│
├── 1_raw_data/               All the recordings (50 recordings of 8 cultures)
├── software_outputs/         Results of the MATLAB analysis software of the laboratory
│   ├── GTE/                  Effective connectivity networks and their metrics
│   └── VisualizeFront/       Global network activity (GNA) of the whole network
│
├── 2_screening/              Valid recordings                          -> recordings.csv
├── 3_window/                 Coincidence window and chamber threshold  -> Fig. SF3
├── 4_culture_selection/      Coupled cultures                          -> Table ST1 (cultures.csv)
├── 5_analysis_data/          Recordings of the 7 coupled cultures (36), copied by step 4
├── 6_directionality/         Direction of propagation                  -> Figs. 5e, 5f, 5g
├── 7_example_culture/        Example culture C1, DIV 17                -> Figs. 5a–d
├── 8_topology/               Topology of the GTE networks              -> Fig. 6c, Table ST2
├── 9_raster_gna_example/     Raster and GNA of C1, DIV 21              -> Fig. 2b
└── 10_in_silico/             Computational model (notebook)            -> Figs. 3a, 3b, 3d, 3e, 5h, 5i, 5j
```

## Data

- `spikesData.csv`: ROI and time (s) of every spike, exported from Netcal.
  `ROI_centers_everything.csv`: position of every ROI.
- The 400 ROIs form a 20 × 20 grid: ROIs 1–200 are one chamber (A) and ROIs 201–400 the other (B).
  The source is the chamber at the wide end of the funnel-shaped microchannels; it is given for
  every recording in `5_analysis_data/index.csv`.
- Cultures are named by their date. In the article: C1 = 15_05_1, C2 = 15_05_2, C3 = 30_04_2,
  C4 = 20_02, C5 = 30_04_3, C6 = 22_05, C7 = 30_04_1 and C8 = 20_04 (not coupled, excluded).

## The steps

- **Chamber bursts** (`common/bursts.py`, used by all the steps). The chamber GNA is the
  fraction of the active ROIs of each chamber that fire within 0.5 s; a chamber burst is an
  episode in which it exceeds 0.1, and its time is the time of its peak.
- **2 – Screening.** Excludes the recordings made after DIV 24, those without bursts in one of
  the chambers, and two faulty recordings (20_04 DIV 7: wrong frame rate; 30_04_1 DIV 11:
  abnormal activity).
- **3 – Window.** Compares the time between the bursts of the two chambers with what is expected
  by chance, and sets the coincidence window to 1 s. `threshold_check.py` compares the burst
  thresholds 0.1 and 0.2.
- **4 – Culture selection.** Keeps the cultures whose chambers have more coincident bursts than
  expected by chance (p < 0.05).
- **6 – Directionality.** Bursts of the two chambers closer than 1 s are one propagation, and the
  chamber that peaks first is its origin (S->T or T->S); if both peak at the same time step the
  propagation is simultaneous and has no direction; bursts without a partner are local.
  The counts of each culture give the directionality index, the recruitment score and a binomial
  test.
- **7 – Example culture.** The same analysis for one recording, plus the initiation site of
  every burst.
- **8 – Topology.** Reads the GTE networks of `software_outputs/GTE`, keeps the reliable ones,
  and compares their modularity, small-worldness and global efficiency with random networks. It
  also compares the communities with the two chambers and computes the reciprocity of the
  connections.
- **9 – Raster and GNA.** Raster of one recording and its GNA, taken from
  `software_outputs/VisualizeFront`.
- **10 – In silico model.** Everything is explained in the notebook. `in_silico_networks.csv`
  contains the values of each of the 10 simulated networks.

## How to run

Install the libraries once:

```
python3 -m pip install -r requirements.txt
```

Then run the steps in order, from the main folder:

```
cd 2_screening             && python3 screening.py
cd ../3_window             && python3 window.py && python3 threshold_check.py
cd ../4_culture_selection  && python3 selection.py
cd ../6_directionality     && python3 directionality.py
cd ../7_example_culture    && python3 example_culture.py
cd ../8_topology           && python3 topology.py
cd ../9_raster_gna_example && python3 raster_gna.py
cd ../10_in_silico         && jupyter nbconvert --to notebook --execute --inplace in_silico.ipynb
```

Each script prints the numbers reported in the article and saves its figures as PNG (300 dpi)
and PDF.

## License

MIT, see `LICENSE`.
