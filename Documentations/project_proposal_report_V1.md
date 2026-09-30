# NATIONAL UNIVERSITY OF SINGAPORE
## Institute of Systems Science
### Graduate Certificate in Pattern Recognition Systems
**Practice Module Project Proposal**

# AValidate: Intelligent Audio-Visual Source Matching

**Canvas Group Number:** 25

**Team Members:**

| Team Member | NUS Student ID |
| :--- | :--- |
| Veeramani Mahadevan Venkata Ganapathi Subramanian | A0355739M |
| Gupta Swaraj | |
| Ramia Kasthurirangam Pandurengan Ghanshyam | |

> **Proposal Note:** The proposed architecture, datasets, model configurations and team responsibilities represent the current design baseline and may be refined during implementation based on experimental findings, data availability and technical feasibility.

---

## 1. Problem, Motivation & Market Context

Real-world video understanding becomes unreliable when a visible object or event is assumed to be the source of the sound simply because the two occur together. A dog may be visible while a siren is heard, a person may appear while a dog bark dominates the audio, or an off-screen source may produce the sound. A useful intelligent system therefore needs to reason across both modalities rather than classify audio and video independently.

Existing audio classification and visual recognition systems can identify what is heard or seen, but independent predictions do not directly answer whether the visible source actually corresponds to the dominant audio. **AValidate** therefore focuses on audio-visual correspondence, learning aligned audio and visual representations and estimating whether they describe the same event at the same time.

The project investigates the following question:

> *Can synchronized audio and visual embeddings be fused to reliably verify whether the visible event is the source of the observed sound?*

The MVP will use the **AVE (Audio-Visual Event)** dataset as the primary source because it provides 4,143 ten-second audio-video clips across 28 event categories with per-second event/background annotations. **VGGSound** may be used as a scale-up or fallback source for selected overlapping classes, while **VGG-SS** can support optional qualitative sound-source localization evaluation.

### 1.1 Objectives and Scope

The MVP will:
* Extract synchronized audio and video information from each input clip;
* Encode audio using a pretrained BEATs branch and encode visual frames;
* Align the two modalities at approximately one-second temporal resolution;
* Predict the event class and whether an event is both audible and visually present;
* Learn a cross-modal correspondence score for **MATCH / MISMATCH** verification;
* Evaluate positive pairs and deliberately mismatched audio-video pairs; and
* Present event confidence, temporal evidence and the final correspondence result in a simple interface.

Representative cases include *dog video + barking* (expected strong correspondence), *car + car horn* (strong correspondence), *dog video + siren or car horn* (mismatch), and *human/person video + dog bark* (low or uncertain correspondence rather than logical impossibility). A real-time webcam + microphone mode is treated as a final demonstration enhancement after the offline MVP is stable.

---

## 2. Proposed System Design

The system separates modality-specific representation learning from cross-modal verification. The audio and visual branches first produce embeddings independently; these embeddings are then temporally aligned and fused so that the final decision depends on their relationship rather than on either modality alone.

```
+-------------------------------------------------------------------+
|                            Input Video                            |
|                     (10 s clip / live stream)                     |
+---------------------------------+---------------------------------+
                                  |
        +-------------------------+-------------------------+
        |                                                   |
        v                                                   v
+---------------+                                   +---------------+
| Audio Branch  |                                   | Visual Branch |
| Log-Mel /     |                                   | Frames /      |
| BEATs + LoRA  |                                   | CLIP ViT-B/16 |
+-------+-------+                                   +-------+-------+
        |                                                   |
        +-------------------------+-------------------------+
                                  |
                                  v
                     +--------------------------+
                     |    Aligned Embeddings    |
                     | (1 fps temporal alignm.) |
                     +------------+-------------+
                                  |
                                  v
                     +--------------------------+
                     |          Fusion          | ===> [ Optional Upgrade: ]
                     |       Concat + MLP       |      [ Cross-Modal Attn. ]
                     +------------+-------------+
                                  |
        +-------------------------+-------------------------+
        |                         |                         |
        v                         v                         v
+---------------+       +-------------------+       +---------------+
|    Head 1     |       |      Head 2       |       |    Head 3     |
|  Event Class  |       | Audible + Visible |       | Corres. Score |
|               |       | Per-sec Presence  |       | MATCH/MISMATCH|
+-------+-------+       +---------+---------+       +-------+-------+
        |                         |                         |
        +-------------------------+-------------------------+
                                  |
                                  v
                     +--------------------------+
                     |    Hugging Face Space    |
                     |    Gradio Demo Interface |
                     | Result+Confidence+Timeln |
                     +--------------------------+
```
*Figure 1. Proposed system architecture. Audio and visual features are extracted from the same input clip, temporally aligned, fused, and used to predict the event, per-second audio-visual presence, and a final match/mismatch correspondence score. The outputs are presented through a Gradio-based Hugging Face Space.*

* **Audio branch:** A lightweight log-Mel CNN provides a baseline. The main model uses pretrained BEATs audio representations with parameter-efficient LoRA adaptation. The branch produces audio embeddings and event evidence from each temporal segment.
* **Visual branch:** Video frames are sampled at aligned timestamps and encoded using CLIP ViT-B/16 with LoRA adaptation. An optional COCO object detector may be used only as a visualization aid to show visible objects; the correspondence decision is learned from the audio-visual representations.
* **Fusion and prediction heads:** The first implementation uses late fusion by concatenating aligned audio and visual embeddings followed by an MLP. If time permits, this is compared with cross-modal attention. Three outputs are produced: event class, per-second audible-and-visible presence, and a learned audio-video correspondence score.
* **Pair construction:** Original synchronized clips provide positive examples. Negative examples are created by swapping audio between clips from different event classes while preserving clip length and alignment. This explicitly teaches the model the difference between merely co-occurring modalities and semantically corresponding sound sources.

---

## 3. Data & Knowledge

| Source | Use |
| :--- | :--- |
| **AVE (Audio-Visual Event)** | Primary MVP dataset: 4,143 ten-second clips, 28 categories, per-second event/background labels. |
| **Generated Positive / Negative Pairs** | Train and evaluate correspondence using synchronized positives and audio-swapped mismatches. |
| **VGGSound** | Optional scale-up/fallback source using a controlled subset of overlapping audio-visual event classes. |
| **VGG-SS** | Optional qualitative/evaluation resource for visual sound-source localization. |

The team will construct a reproducible preprocessing pipeline that extracts mono audio, samples aligned visual frames, maps class labels, creates correspondence pairs and stores train/validation/test splits. To prevent leakage, negative pairs will be generated within the appropriate split rather than mixing content across training and evaluation partitions.

Because some real scenes are inherently ambiguous, the final system will not interpret a low correspondence score as physical impossibility. For example, a visible person could imitate a dog bark while an off-screen source may also dominate the sound. The interface will therefore report confidence and can use Strong / Weak / Uncertain bands rather than presenting every case as an absolute statement.

---

## 4. Evaluation & Project Execution

| Experiment | Comparison | Measures |
| :--- | :--- | :--- |
| **Audio event recognition** | Log-Mel CNN vs. BEATs + LoRA | Accuracy, Macro-F1, Confusion Matrix |
| **Fusion / correspondence** | Late Fusion vs. Cross-Modal Attention | Accuracy, F1, ROC-AUC, Precision/Recall |
| **Temporal AV verification** | Positive Pairs vs. Audio-Swapped Negative Pairs | Per-second Accuracy/F1, Correspondence Score Calibration |

Evaluation will include both standard quantitative metrics and curated scenario tests such as dog+bark, car+horn, dog+siren and human+dog-bark. Robustness tests may additionally vary audio level, background noise and partial visual obstruction to observe how correspondence confidence changes.

### 4.1 Team Responsibilities

| Member | Responsibilities |
| :--- | :--- |
| **Member 1** | Audio pipeline: log-Mel baseline, BEATs + LoRA, fusion components, output heads and shared inference function |
| **Member 2** | Vision pipeline: frame extraction, CLIP + LoRA, visual baseline/robustness experiments and Streamlit front end. |
| **Member 3** | AVE preprocessing, positive/negative pair generator, evaluation pipeline, experiment tracking, results packaging and demo preparation. |

Integration and end-to-end evaluation will be shared. The components will expose a common inference interface so that a single input video can return the event prediction, confidence values, temporal evidence and final audio-visual match score.

---

## 5. Expected Outcome

GC2 will deliver a runnable audio-visual verification system capable of accepting a video clip, identifying the dominant event information in the audio and visual streams, and estimating whether the two modalities correspond.

The expected technical contribution is the integration of:

$$\text{Audio Representation} \longrightarrow \text{Visual Representation} \longrightarrow \text{Temporal Alignment} \longrightarrow \text{Cross-Modal Fusion} \longrightarrow \text{Correspondence Verification}$$

Rather than treating audio recognition and visual recognition as two unrelated tasks, the project learns the relationship between them. The final output will include the predicted event, confidence information, a temporal presence timeline and a correspondence decision such as Strong Match, Weak/Uncertain, or Mismatch.

After the core offline pipeline is completed, the same inference flow can be connected to a webcam and microphone for a live demonstration. This real-time mode is an enhancement rather than a dependency of the MVP.

---

## References

1. Tian, Y. et al., “Audio-Visual Event Localization in Unconstrained Videos,” *European Conference on Computer Vision (ECCV)*, 2018.
2. Chen, H. et al., “VGGSound: A Large-scale Audio-Visual Dataset,” *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2020.
3. Chen, H. et al., “Localizing Visual Sounds the Hard Way,” *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021.
4. Chen, S. et al., “BEATs: Audio Pre-Training with Acoustic Tokenizers,” 2022.
5. Radford, A. et al., “Learning Transferable Visual Models From Natural Language Supervision,” *International Conference on Machine Learning (ICML)*, 2021.

---

**AI Tool Declaration:** We used ChatGPT to help review and refine the proposed system architecture, structure the project proposal, improve technical expression, and assist in preparing the architecture diagram. All model choices, dataset selections, implementation decisions, experiments, revisions and final submitted content will be reviewed and approved by the project team.