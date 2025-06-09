[![DeepSource](https://app.deepsource.com/gh/soheilabadifard/DynED.svg/?label=active+issues&show_trend=true&token=o4DHpyfp0usOBhkJM5DPUbyL)](https://app.deepsource.com/gh/soheilabadifard/DynED/?ref=repository-badge)

# DynED: Dynamic Ensemble Diversification in Data Stream Classification

[cite_start]This is the official implementation of the paper "DynED: Dynamic Ensemble Diversification in Data Stream Classification" [cite: 10][cite_start], published at CIKM 2023.

[cite_start]DynED is a novel ensemble construction and maintenance approach for data stream classification that dynamically balances the diversity and prediction accuracy of its components. [cite_start]The core challenge in data stream environments is handling disruptive changes in the data distribution, known as concept drift. [cite_start]DynED addresses this by using the Maximal Marginal Relevance (MMR) concept to dynamically adjust the ensemble's diversity—increasing it to adapt during concept drifts and decreasing it to maximize accuracy in stable periods.

**Authors:** Soheil Abadifard, Sepehr Bakhshi, Sanaz Gheibuni, and Fazli Can.

---

## Key Features

* [cite_start]**Dynamic Diversity Adjustment**: DynED dynamically adjusts its diversity parameter based on the intensity of accuracy changes, allowing it to adapt to severe drifts without manual tuning.
* [cite_start]**MMR-based Component Selection**: Utilizes a modified Maximal Marginal Relevance (MMR) method to prune redundant or ineffective components, ensuring the ensemble is both diverse and accurate.
* [cite_start]**Concept Drift Adaptation**: Specifically designed to handle the challenges of concept drift in evolving data streams [cite: 38][cite_start], outperforming baseline methods in various drift scenarios.
* [cite_start]**Proven Performance**: Experimental results on 15 datasets show that DynED achieves a higher average mean accuracy compared to five state-of-the-art baselines.

---
<details>
<summary><b>How it Works</b></summary>

[cite_start]DynED's architecture is built on a three-stage process to construct and maintain the ensemble structure, as illustrated in the paper.

1.  **Stage 1: Prediction & Training**
    * [cite_start]The set of active components predicts the label of new data samples using majority voting.
    * These components are then trained on the new samples in an online fashion.

2.  **Stage 2: Drift Detection & Adaptation**
    * [cite_start]The ADWIN drift detector is used to monitor the predictions for concept drift.
    * [cite_start]If a drift is detected, a new component is trained on the most recent data and added to a "reserved pool" of components.
    * [cite_start]The diversity parameter (`λ`) is dynamically updated based on the rate of change in the ensemble's accuracy, preparing it for the selection stage.

3.  **Stage 3: Component Selection**
    * [cite_start]When triggered, this stage combines the active and reserved components and prunes the pool to a maximum size.
    * [cite_start]Components are first clustered into two groups based on their prediction errors on recent data.
    * [cite_start]Finally, the adapted MMR method is used to select a new set of high-performing, diverse components to become the active ensemble.

</details>

<br>

## Installation

1.  [cite_start]**Prerequisites**: DynED requires **Python 3.8**.
2.  [cite_start]**Dependencies**: The core dependency is the `scikit-multiflow` library. The last version of `scikit-multiflow` is compatible with specific versions of Numpy and Pandas. Please follow their official installation instructions.

    ```bash
    pip install -U scikit-multiflow
    ```
    For more details, visit the [scikit-multiflow documentation](https://scikit-multiflow.readthedocs.io/en/stable/index.html).

## How to Run

1.  **Get the Data**: The datasets used for evaluation are available in the `/Dataset` folder.
2.  **Configure the Script**:
    * Open the `DynED.py` file.
    * Locate the following line:
        ```python
        stream = FileStream("Put the full address and name of the dataset here.")
        ```
    * Replace the placeholder string with the full path to the dataset file you wish to evaluate. For example:
        ```python
        stream = FileStream("Dataset/poker.arff")
        ```
3.  **Execute the Script**:
    * Run the Python script from your terminal:
        ```bash
        python DynED.py
        ```
    * The script will output the final mean accuracy of the DynED model on the provided dataset.

## Baseline Experiments

[cite_start]The paper evaluates DynED against five state-of-the-art baselines: **LevBag**, **SAM-KNN**, **ARF**, **SRP**, and **KUE**.

The scripts used to run these baseline experiments using the [MOA framework](https://moa.cms.waikato.ac.nz/) can be found in the `/scripts` directory. Please see the `README.md` within that directory for more information.

## Acknowledgments

This study is partially supported by TÜBİTAK grant no. [cite_start]122E271.

## Citation

If you use this work in your research, please cite the original paper:

```bibtex
@inproceedings{abadifard2023dyned,
  author = {Abadifard, Soheil and Bakhshi, Sepehr and Gheibuni, Sanaz and Can, Fazli},
  title = {DynED: Dynamic Ensemble Diversification in Data Stream Classification},
  year = {2023},
  isbn = {9798400701245},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {[https://doi.org/10.1145/3583780.3615266](https://doi.org/10.1145/3583780.3615266)},
  doi = {10.1145/3583780.3615266},
  booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages = {3707–3711},
  numpages = {5},
  keywords = {ensemble pruning, maximal marginal relevance, data stream classification, concept drift, diversity adjustment, ensemble learning},
  location = {Birmingham, United Kingdom},
  series = {CIKM '23}
}
