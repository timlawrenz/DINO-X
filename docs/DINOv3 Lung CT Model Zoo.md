# **Architectural Foundations of DINOv3 and the Evolution of Vision Foundation Models for Pulmonary Imaging**

The landscape of medical computer vision is currently undergoing a radical transformation, moving away from task-specific, supervised models toward general-purpose vision foundation models (VFMs). This paradigm shift is most acutely felt in the analysis of lung computed tomography (CT) scans, where the complexity of three-dimensional anatomical structures and the high cost of expert annotation have historically limited the scalability of deep learning solutions. The release of Meta AI’s DINOv3 in August 2025 represents a critical milestone in this evolution.1 As a self-supervised vision transformer (ViT) pre-trained on an unprecedented scale of 1.7 billion images, DINOv3 introduces technical innovations—most notably Gram Anchoring—that resolve the long-standing problem of dense feature degradation in large-scale models.3 For the medical research community, the challenge now lies in effectively adapting these multi-billion-parameter backbones to the clinical domain and establishing a reproducible "model zoo" that can serve diverse diagnostic needs across varied institutional infrastructures.5

## **The Technical Architecture of DINOv3: From Global to Dense Representations**

DINOv3 (Distillation with No Labels v3) is engineered to learn robust, universal visual representations without the need for human-provided labels.1 It builds upon the teacher-student self-distillation framework established by its predecessors, DINO and DINOv2, but scales the parameter count to 7 billion and the training data by a factor of twelve.1 The core of the DINOv3 training process involves a student model that learns to match the target representations provided by a momentum-updated teacher model.3 This process is facilitated through a multi-crop strategy where the student and teacher are exposed to different augmented views of the same image, encouraging the model to capture invariant features that remain consistent across transformations such as cropping, color shifts, and geometric blurs.3

### **Gram Anchoring and the Stability of Local Features**

The most significant technical innovation in DINOv3 is the introduction of Gram Anchoring, a regularization technique designed to stabilize dense feature learning during long training runs.3 In previous large-scale self-supervised models, researchers observed a "feature collapse" phenomenon: while global image-level descriptors continued to improve, patch-level (dense) features tended to lose their spatial specificity.3 Patches that should remain distinct began to align too closely with the global token, resulting in noisy and incoherent feature maps that degraded performance on downstream tasks like segmentation and depth estimation.3

Gram Anchoring addresses this by operating on the Gram matrix, which encodes the second-order correlations and pairwise similarities between patches within an image.3 During the training process, the student model’s Gram matrix is encouraged to remain close to that of an earlier, more stable "Gram teacher" checkpoint.3 This is mathematically represented by the Gram loss ![][image1]:

![][image2]  
10

where ![][image3] represents the patch features of the student and ![][image4] represents those of the Gram teacher.10 By anchoring these relative structures, DINOv3 ensures that local features retain their semantic sharpness, allowing for the precise delineation of small pulmonary nodules and fine vascular structures in lung CT volumes.3

### **Multi-Objective Optimization for Unified Vision**

DINOv3 does not rely on a single objective but rather a composite loss function that balances global and local representation learning.4 The total training objective combines the DINO image-level distillation loss, the iBOT patch-level reconstruction loss (which leverages masked image modeling), and the Koleo uniformity regularizer to ensure the embedding space is utilized efficiently.4

| Loss Component | Objective Type | Technical Function |
| :---- | :---- | :---- |
| **![][image5]** | Image-level | Aligning global representations of local and global crops for invariant features. |
| ![][image6] | Patch-level | Matching teacher representations for masked tokens to learn part-whole relationships. |
| ![][image7] | Regularization | Maximizing the entropy of features to prevent representational collapse. |
| ![][image1] | Dense-level | Stabilizing patch-wise similarity maps to maintain spatial granularity. |

Table 1: The multi-objective loss framework of DINOv3.4

The integration of these objectives, coupled with architectural updates such as Rotary Positional Embeddings (RoPE), allows DINOv3 to handle images of varying resolutions and aspect ratios—a critical requirement for medical imaging where CT reconstruction kernels and slice thicknesses vary between scanner manufacturers.1

## **Common Approaches for Lung CT Scan Analysis Using DINOv3**

The application of DINOv3 to lung CT analysis follows several distinct pathways, ranging from zero-shot inference to domain-specific fine-tuning. Because DINOv3 features are "emergent"—meaning they possess semantic and spatial properties without explicit supervision—they can often be used effectively with minimal task-specific training.1

### **Frozen Feature Extraction and Linear Probing**

The most efficient way to leverage DINOv3 is as a universal frozen backbone.1 In this approach, the core weights of the 7 billion parameter model remain fixed, and only a lightweight task-specific head (such as a linear layer or a shallow MLP) is trained on top of the extracted features.3 This method is particularly effective for classification tasks such as malignancy risk estimation or the identification of interstitial lung disease.16 Benchmarking reports indicate that DINOv3-L (Large) achieves an AUC of 0.7865 on the NIH-14 chest X-ray dataset, outperforming the medical-specific BiomedCLIP model despite having no prior exposure to radiological data.17

### **Domain-Adaptive Pre-training: The MedDINOv3 Framework**

While off-the-shelf features are surprisingly strong, the "domain gap" between natural images (RGB) and medical scans (grayscale, high-dynamic-range CT) remains a bottleneck for high-precision tasks.3 To address this, the MedDINOv3 framework introduces a three-stage domain-adaptive pre-training recipe.13 In this workflow, a DINOv3-pretrained backbone is further trained on a curated collection known as CT-3M, which comprises nearly 4 million axial CT slices.13

The MedDINOv3 approach specifically revisits the Vision Transformer architecture for medical segmentation by introducing multi-scale token aggregation.13 Instead of relying solely on the final layer’s outputs, the model reuses patch tokens from intermediate transformer blocks, providing the decoder with hierarchical spatial context that is essential for segmenting complex anatomical regions like the mediastinum or the pulmonary hila.13

### **Feature Disentanglement (DINOv3-FD)**

Advanced adaptation strategies like DINOv3-FD focus on the quality of the latent space.21 Pre-trained features often interleave task-relevant diagnostic semantics with task-irrelevant noise, such as scanner artifacts or patient-specific anatomical variations that do not contribute to a diagnosis.21 DINOv3-FD utilizes a dual-stream adapter that decomposes features into a task-relevant subspace and a task-irrelevant subspace.21 An orthogonality loss reinforces the mutual independence of these subspaces, allowing the diagnostic stream to focus exclusively on features relevant to lung nodule malignancy or disease progression.21

## **Developing a Model Zoo: Reproducible Datasets and Scaling**

A "model zoo" is defined as a structured collection of pre-trained models that provide scalable solutions for diverse resource constraints and deployment scenarios.5 To develop a novel model zoo for lung CT research, one must combine high-quality public datasets with a systematic scaling strategy that spans from ultra-lightweight models to flagship backbones.5

### **Strategic Selection of Public Datasets**

Reproducibility in medical AI is dependent on the availability of open-access data with standardized annotations.24 For lung CT model development, several key datasets provide the necessary scale and diversity.

| Dataset | Modality | Scale | Clinical Significance |
| :---- | :---- | :---- | :---- |
| **NLST** | Low-dose CT | \>5,000 scans | Standard for lung cancer screening and longitudinal risk studies. |
| **LUNA25** | Chest CT | 4,000+ scans | Benchmarking nodule detection and malignancy risk estimation. |
| **CT-3M** | Multi-region CT | 3.87M slices | Large-scale pre-training for general CT feature extraction. |
| **DLCST** | Low-dose CT | \>2,000 scans | External validation for screening algorithms in diverse populations. |
| **Lung-PET-CT-Dx** | PET / CT | 355+ cases | Multimodal fusion for tumor subtyping and metabolic analysis. |

Table 2: High-impact public datasets for lung CT foundation model development.20

The National Lung Screening Trial (NLST) is particularly critical, as it offers not only imaging but also rich clinical metadata and longitudinal outcomes, allowing models to learn phenotype-level representations.27 The LUNA25 challenge, a new grand challenge launching in 2025, provides a standardized evaluation framework with over 4,000 carefully annotated exams, designed to estimate both AI and radiologist performance at lung nodule malignancy risk estimation.30

### **Scaling the Model Family: Teacher-Student Distillation**

A professional model zoo should include a range of model sizes to accommodate different clinical workflows, from high-fidelity research workstations to resource-constrained mobile health units.6 DINOv3 facilitates this through a single-teacher, multi-student distillation pipeline.8

1. **The Flagship (Teacher):** The 7B-parameter DINOv3 (ViT-7B) serves as the primary feature extractor and the teacher model for the rest of the zoo.1  
2. **Standard Backbones (Large/Base/Small):** These variants (e.g., ViT-L with 300M parameters, ViT-B with 86M, and ViT-S with 21M) offer a range of accuracy-latency trade-offs for general diagnostic tasks.8  
3. **Efficient Backbones (ConvNeXt):** For real-time applications, the DINOv3 suite includes distilled ConvNeXt models (ranging from 29M to 198M parameters) that are optimized for speed-constrained environments.6  
4. **Ultra-Lightweight Variants (Nano/Pico):** Emerging architectures like MambaOut-Femto or DEIMv2-Pico (sub-5M parameters) redefine the efficiency frontier, achieving competitive results with minimal memory footprints.22

Scaling laws in the medical domain suggest that while larger models (ViT-L) generally provide better features, they do not consistently follow the same linear improvement seen in natural images.17 For instance, on the RSNA Pneumonia task, performance often plateaus or even declines if input resolution is not carefully matched to the model’s patch size.18

## **Leveraging Infrastructure for Publication and Distribution**

The value of a model zoo is intrinsically tied to its accessibility and the ease with which other researchers can reproduce its results.26 Existing infrastructure platforms such as Hugging Face, MONAI, and MHub.ai provide the necessary tools for professional dissemination.7

### **Hugging Face Hub and PyTorchModelHubMixin**

The Hugging Face Hub serves as the primary repository for modern AI weights.36 To integrate a model zoo effectively, researchers should leverage the PyTorchModelHubMixin class, which allows for seamless model loading via the from\_pretrained method.36

Professional model cards are essential for clinical transparency.20 A high-quality medical model card on Hugging Face should include:

* **YAML Metadata:** Specifically identifying task: image-segmentation and modality: CT to enable discovery via the hub’s filtering system.36  
* **Bias and Risks:** Explicitly stating the populations and scanner types (e.g., GE vs. Siemens) included in the training data to prevent "hidden" dataset shifts.20  
* **Inference Examples:** Providing clear code snippets for processing 3D DICOM volumes, which typically require conversion to NIfTI or 3D tensors before input.20

### **Containerization with MHub.ai and MONAI**

To overcome the friction of installation and dependency conflicts, medical models should be containerized.35 MHub.ai provides a standardized platform that packages AI models into Docker containers with built-in DICOM compatibility.35 Each model in the zoo should define an explicit end-to-end workflow—from raw DICOM ingestion to the generation of DICOM-SEG or RTStruct outputs—ensuring that clinical users can execute the models without local code modifications.35

The MONAI Model Zoo further complements this by hosting models in the "MONAI Bundle" format, which encapsulates not only the weights but also the specific transforms (e.g., intensity normalization, spacing resampling) required for the model to operate correctly.7

## **3D Adaptation: The Dimensionality Challenge**

Lung CT is inherently volumetric, while DINOv3 is primarily a 2D vision transformer.1 Bridging this gap is a central theme in recent research, with two primary methodologies emerging: pseudo-3D stacking and native 3D pre-training.14

### **Pseudo-3D and the 2.5D Paradigm**

In the pseudo-3D approach, DINOv3 is treated as a 2D encoder that processes each axial slice independently.16 The resulting 2D feature maps are then stacked to form a pseudo-3D feature volume, which is fed into a 3D decoder.16 This "2.5D" strategy allows researchers to benefit from the rich semantic features learned from billions of natural images while still incorporating some degree of longitudinal context along the z-axis.14 For many classification tasks, this approach is highly effective; models like TAP-S-2.5D often perform competitively with or better than native 3D models on lung cancer subtyping.14

### **Native 3D Pre-training: TAP-CT and 3DINO**

For tasks requiring absolute spatial precision, such as surgical planning or radiotherapy dosing, native 3D pre-training is required.43 The TAP-CT (Task-Agnostic Pretraining of CT) and 3DINO-ViT frameworks adapt the DINO/ViT architecture for volumetric data by using 3D patch embeddings (cuboids rather than squares) and 3D positional encodings.43

| Architecture | Patch Dimension | Positional Encoding | Data Volume |
| :---- | :---- | :---- | :---- |
| **DINOv3 (Standard)** | **![][image8]** (2D) | RoPE (2D) | 1.7B Natural Images |
| **TAP-CT (3D)** | **![][image9]** (3D) | Depth-aware 3D | 105K CT Volumes |
| **MedDINOv3 (2.5D)** | **![][image10]** (2D) | Standard ViT | 3.8M CT Slices |

Table 3: Comparison of dimensionality adaptation strategies in recent CT foundation models.4

While 3D pre-training is computationally intensive—the TAP-B-3D model consumed 2.55 MWh during its training phase—it produces frozen representations that generalize exceptionally well across unseen organs and modalities, establishing a new "gold standard" for task-agnostic medical vision encoders.14

## **Publishing as a Novel Scientific Contribution**

To elevate a model zoo from a technical repository to a novel scientific contribution, the researcher must adhere to the rigorous standards of journals such as *Scientific Data* or *Scientific Reports*.34 This involves providing not just the code and weights, but a "Machine Learning Model Descriptor" that documents the entire lifecycle of the model.34

### **Adherence to FAIR Principles**

The publication must demonstrate that the models are FAIR-compliant (Findable, Accessible, Interoperable, and Reusable).34 This requires:

1. **Unique Identity:** Each model version must have a persistent identifier (DOI) and clear versioning (e.g., using GitHub Releases).7  
2. **Structured Metadata:** All images, masks, and clinical descriptors should adhere to a DICOM-based metadata structure, enabling ingestion into repositories like the National Cancer Institute's Image Data Commons (IDC).42  
3. **Transparent Data Cleaning:** The manuscript must describe the protocol used for data de-identification, outlier removal, and class balancing (e.g., handling the inherent imbalanced nature of malignant vs. benign nodules).48

### **Standardized Benchmarking**

A novel contribution is validated through its performance on independent, external datasets.31 For lung CT, this means reporting metrics not only on internal test sets but also on multinational cohorts from different acquisition eras.51 The use of the LUNA25 benchmark as a "living review" allows researchers to see how their model zoo ranks against commercial AI solutions and expert radiologists across different operating points.31

## **Ethics, Safety, and Clinical Reality**

The clinical implementation of a large-scale model zoo introduces unique ethical considerations.40 As foundation models like DINOv3 become increasingly trusted, there is a risk of "automation bias," where physicians may overlook AI errors due to the model's high general performance.40

### **Explainability and Trust**

Models in the zoo should prioritize explainability, either through mechanistic interpretability (e.g., identifying biological factors like genetic mutations) or post-hoc methods like SHAP or DeepLIFT.53 Recent efforts to align AI decision-making with clinical reasoning—such as predicting pathology-related visual attributes (shape, texture, spiculation) alongside the final diagnosis—enhance clinician trust and usability.55

### **Monitoring Data and Model Drift**

The data used for training foundation models and the models themselves can diverge over time due to changes in image acquisition protocols or disease prevalence.40 A robust model zoo must therefore include tools for continuous monitoring of model outputs and OOD detection to alert users when a model is being applied to data that differs significantly from its training distribution.39

## **Comprehensive Synthesis and Recommendations**

The development and publication of a DINOv3-based model zoo for lung CT analysis is a multi-faceted endeavor that sits at the intersection of large-scale engineering and rigorous clinical science. DINOv3, through its innovations in Gram Anchoring and self-distillation, provides a universal vision backbone that can serve as the cornerstone of this initiative.1 By scaling these backbones through distillation, researchers can provide efficient solutions for every point on the diagnostic spectrum, from rapid screening to complex oncological staging.5

For the contribution to be recognized as novel, it must be grounded in the principles of reproducibility and open science. Utilizing the Hugging Face Hub for distribution, MHub.ai for containerization, and the LUNA25 challenge for benchmarking ensures that the model zoo is not just a collection of files, but a functional, transparent ecosystem.30 Ultimately, the goal is to lower the barrier to clinical translation, enabling side-by-side benchmarking and accelerating the development of reliable AI tools that can truly assist radiologists in the global fight against lung cancer.30

The future of this field lies in multimodal convergence, where vision foundation models are paired with clinical narratives and longitudinal data to provide a holistic, patient-centric understanding.27 As researchers begin to explore 4D CT analysis and generative data augmentation, the foundational representations provided by DINOv3 will remain an essential anchor for the next decade of medical imaging innovation.10

#### **Works cited**

1. DINOv3 Explained: The Game-Changing Vision Transformer That's Redefining Computer Vision | by Abhishek Selokar | Medium, accessed February 1, 2026, [https://medium.com/@imabhi1216/dinov3-explained-the-game-changing-vision-transformer-thats-redefining-computer-vision-cd63646141e6](https://medium.com/@imabhi1216/dinov3-explained-the-game-changing-vision-transformer-thats-redefining-computer-vision-cd63646141e6)  
2. Meta Releases DINOv3: 7 Billion Parameter Visual Model Begins a New Era in Computer Vision \- Oreate AI Blog, accessed February 1, 2026, [https://www.oreateai.com/blog/meta-releases-dinov3-7-billion-parameter-visual-model-begins-a-new-era-in-computer-vision/12e13ccfd1041168dbfee6cf0562eb57](https://www.oreateai.com/blog/meta-releases-dinov3-7-billion-parameter-visual-model-begins-a-new-era-in-computer-vision/12e13ccfd1041168dbfee6cf0562eb57)  
3. DINOv3 Explained: Scaling Self-Supervised Vision Transformers | Encord, accessed February 1, 2026, [https://encord.com/blog/dinov3-explained-scaling-self-supervised-vision-tr/](https://encord.com/blog/dinov3-explained-scaling-self-supervised-vision-tr/)  
4. Paper Review: DINOv3 \- Andrey Lukyanenko, accessed February 1, 2026, [https://andlukyane.com/blog/paper-review-dinov3](https://andlukyane.com/blog/paper-review-dinov3)  
5. \[2508.10104\] DINOv3 \- arXiv, accessed February 1, 2026, [https://arxiv.org/abs/2508.10104](https://arxiv.org/abs/2508.10104)  
6. DINOv3 \- arXiv, accessed February 1, 2026, [https://arxiv.org/html/2508.10104v1](https://arxiv.org/html/2508.10104v1)  
7. Project-MONAI/model-zoo \- GitHub, accessed February 1, 2026, [https://github.com/Project-MONAI/model-zoo](https://github.com/Project-MONAI/model-zoo)  
8. DINOv3 Explained: Technical Deep Dive \- Lightly, accessed February 1, 2026, [https://www.lightly.ai/blog/dinov3](https://www.lightly.ai/blog/dinov3)  
9. DINOv3: A Deep, Practical Overview of Meta's New Self-Supervised Vision Backbone, accessed February 1, 2026, [https://medium.com/@xiaxiami/dinov3-a-deep-practical-overview-of-metas-new-self-supervised-vision-backbone-005311689dbf](https://medium.com/@xiaxiami/dinov3-a-deep-practical-overview-of-metas-new-self-supervised-vision-backbone-005311689dbf)  
10. DINOv3 Vision Foundation Model \- Emergent Mind, accessed February 1, 2026, [https://www.emergentmind.com/topics/dinov3-vision-foundation-model](https://www.emergentmind.com/topics/dinov3-vision-foundation-model)  
11. MedDINOv3: How to adapt vision foundation models for medical image segmentation?, accessed February 1, 2026, [https://www.alphaxiv.org/overview/2509.02379v3](https://www.alphaxiv.org/overview/2509.02379v3)  
12. Dino U-Net: Exploiting High-Fidelity Dense Features from Foundation Models for Medical Image Segmentation \- arXiv, accessed February 1, 2026, [https://arxiv.org/html/2508.20909v1](https://arxiv.org/html/2508.20909v1)  
13. MedDINOv3: How to adapt vision foundation models for medical image segmentation?, accessed February 1, 2026, [https://arxiv.org/html/2509.02379v1](https://arxiv.org/html/2509.02379v1)  
14. \[Literature Review\] TAP-CT: 3D Task-Agnostic Pretraining of Computed Tomography Foundation Models \- Moonlight | AI Colleague for Research Papers, accessed February 1, 2026, [https://www.themoonlight.io/en/review/tap-ct-3d-task-agnostic-pretraining-of-computed-tomography-foundation-models](https://www.themoonlight.io/en/review/tap-ct-3d-task-agnostic-pretraining-of-computed-tomography-foundation-models)  
15. Does DINOv3 Set a New Medical Vision Standard? \- ResearchGate, accessed February 1, 2026, [https://www.researchgate.net/publication/395355707\_Does\_DINOv3\_Set\_a\_New\_Medical\_Vision\_Standard](https://www.researchgate.net/publication/395355707_Does_DINOv3_Set_a_New_Medical_Vision_Standard)  
16. \[Literature Review\] Does DINOv3 Set a New Medical Vision Standard?, accessed February 1, 2026, [https://www.themoonlight.io/en/review/does-dinov3-set-a-new-medical-vision-standard](https://www.themoonlight.io/en/review/does-dinov3-set-a-new-medical-vision-standard)  
17. Does DINOv3 Set a New Medical Vision Standard? A Comprehensive Benchmark on 2D/3D Classification and Segmentation \- arXiv, accessed February 1, 2026, [https://arxiv.org/html/2509.06467v1](https://arxiv.org/html/2509.06467v1)  
18. Does DINOv3 Set a New Medical Vision Standard? Benchmarking 2D and 3D Classification, Segmentation, and Registration \- arXiv, accessed February 1, 2026, [https://arxiv.org/html/2509.06467v3](https://arxiv.org/html/2509.06467v3)  
19. Does DINOv3 Set a New Medical Vision Standard? A Comprehensive Benchmark on 2D/3D Classification and Segmentation \- arXiv, accessed February 1, 2026, [https://arxiv.org/html/2509.06467v2](https://arxiv.org/html/2509.06467v2)  
20. ricklisz123/MedDINOv3-ViTB-16-CT-3M \- Hugging Face, accessed February 1, 2026, [https://huggingface.co/ricklisz123/MedDINOv3-ViTB-16-CT-3M](https://huggingface.co/ricklisz123/MedDINOv3-ViTB-16-CT-3M)  
21. Incentivizing DINOv3 Adaptation for Medical Vision Tasks via Feature Disentanglement \- OpenReview, accessed February 1, 2026, [https://openreview.net/pdf/900aaa3b23f86eb23c1a0ecc9f9b3ee1b1365b30.pdf](https://openreview.net/pdf/900aaa3b23f86eb23c1a0ecc9f9b3ee1b1365b30.pdf)  
22. Evaluation of recent lightweight deep learning architectures for lung cancer CT classification, accessed February 1, 2026, [https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2025.1647701/full](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2025.1647701/full)  
23. Real-Time Object Detection Meets DINOv3 \- arXiv, accessed February 1, 2026, [https://arxiv.org/html/2509.20787v2](https://arxiv.org/html/2509.20787v2)  
24. RIDER-LUNG-CT \- The Cancer Imaging Archive (TCIA), accessed February 1, 2026, [https://www.cancerimagingarchive.net/collection/rider-lung-ct/](https://www.cancerimagingarchive.net/collection/rider-lung-ct/)  
25. (PDF) NodMAISI: Nodule-Oriented Medical AI for Synthetic Imaging \- ResearchGate, accessed February 1, 2026, [https://www.researchgate.net/publication/398978789\_NodMAISI\_Nodule-Oriented\_Medical\_AI\_for\_Synthetic\_Imaging](https://www.researchgate.net/publication/398978789_NodMAISI_Nodule-Oriented_Medical_AI_for_Synthetic_Imaging)  
26. Survey about Barriers and Solutions for Enhancing Computational Reproducibility in Scientific Research | F1000Research, accessed February 1, 2026, [https://f1000research.com/articles/14-1278/pdf](https://f1000research.com/articles/14-1278/pdf)  
27. Foundation Models for Lung Cancer Screening and Prognostics- \- Approved Projects, accessed February 1, 2026, [https://cdas.cancer.gov/approved-projects/5779/](https://cdas.cancer.gov/approved-projects/5779/)  
28. Characterizing the Impact of Training Data on Generalizability: Application in Deep Learning to Estimate Lung Nodule Malignancy Risk | Radiology: Artificial Intelligence \- RSNA Journals, accessed February 1, 2026, [https://pubs.rsna.org/doi/10.1148/ryai.240636](https://pubs.rsna.org/doi/10.1148/ryai.240636)  
29. Lung-PET-CT-Dx: Multimodal Lung Imaging Dataset \- Emergent Mind, accessed February 1, 2026, [https://www.emergentmind.com/topics/lung-pet-ct-dx-dataset](https://www.emergentmind.com/topics/lung-pet-ct-dx-dataset)  
30. Luna25 \- The LUNA25 Challenge \- Grand Challenge, accessed February 1, 2026, [https://luna25.grand-challenge.org/](https://luna25.grand-challenge.org/)  
31. Artificial intelligence in chest imaging \- ESR Connect \- European Society of Radiology, accessed February 1, 2026, [https://connect.myesr.org/?esrc\_course=artificial-intelligence-in-chest-imaging](https://connect.myesr.org/?esrc_course=artificial-intelligence-in-chest-imaging)  
32. Paper page \- AI in Lung Health: Benchmarking Detection and Diagnostic Models Across Multiple CT Scan Datasets \- Hugging Face, accessed February 1, 2026, [https://huggingface.co/papers/2405.04605](https://huggingface.co/papers/2405.04605)  
33. \[2509.06467\] Does DINOv3 Set a New Medical Vision Standard? Benchmarking 2D and 3D Classification, Segmentation, and Registration \- arXiv, accessed February 1, 2026, [https://arxiv.org/abs/2509.06467](https://arxiv.org/abs/2509.06467)  
34. On the Encapsulation of Medical Imaging AI Algorithms \- arXiv, accessed February 1, 2026, [https://arxiv.org/html/2504.21412v2](https://arxiv.org/html/2504.21412v2)  
35. MHub.ai: A Simple, Standardized, and Reproducible Platform for AI Models in Medical Imaging \- ResearchGate, accessed February 1, 2026, [https://www.researchgate.net/publication/399808891\_MHubai\_A\_Simple\_Standardized\_and\_Reproducible\_Platform\_for\_AI\_Models\_in\_Medical\_Imaging](https://www.researchgate.net/publication/399808891_MHubai_A_Simple_Standardized_and_Reproducible_Platform_for_AI_Models_in_Medical_Imaging)  
36. Uploading models \- Hugging Face, accessed February 1, 2026, [https://huggingface.co/docs/hub/models-uploading](https://huggingface.co/docs/hub/models-uploading)  
37. ZJU-AI4H/Hulu-Med-7B \- Hugging Face, accessed February 1, 2026, [https://huggingface.co/ZJU-AI4H/Hulu-Med-7B](https://huggingface.co/ZJU-AI4H/Hulu-Med-7B)  
38. Model Cards \- Hugging Face, accessed February 1, 2026, [https://huggingface.co/docs/hub/model-cards](https://huggingface.co/docs/hub/model-cards)  
39. MY WORK \- Dre Peeters, accessed February 1, 2026, [https://drepeeters.nl/work](https://drepeeters.nl/work)  
40. Foundation Models in Radiology: What, How, Why, and Why Not \- RSNA Journals, accessed February 1, 2026, [https://pubs.rsna.org/doi/10.1148/radiol.240597](https://pubs.rsna.org/doi/10.1148/radiol.240597)  
41. MedDINOv3: How to adapt vision foundation models for medical image segmentation? \- GitHub, accessed February 1, 2026, [https://github.com/ricklisz/MedDINOv3](https://github.com/ricklisz/MedDINOv3)  
42. AI generated annotations for Breast, Brain, Liver, Lungs, and Prostate cancer collections in the National Cancer Institute Imaging Data Commons \- NIH, accessed February 1, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12307902/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12307902/)  
43. arxiv.org, accessed February 1, 2026, [https://arxiv.org/html/2512.00872v1](https://arxiv.org/html/2512.00872v1)  
44. TAP-CT: 3D Task-Agnostic Pretraining of Computed Tomography Foundation Models \- arXiv, accessed February 1, 2026, [https://arxiv.org/abs/2512.00872](https://arxiv.org/abs/2512.00872)  
45. TAP-CT: 3D Task-Agnostic Pretraining of Computed Tomography Foundation Models \- OpenReview, accessed February 1, 2026, [https://openreview.net/pdf/f9d93deb9333e4e0610b33fcf65665c2eacf548b.pdf](https://openreview.net/pdf/f9d93deb9333e4e0610b33fcf65665c2eacf548b.pdf)  
46. 3DINO-ViT | PDF | Image Segmentation \- Scribd, accessed February 1, 2026, [https://www.scribd.com/document/984573326/3DINO-ViT](https://www.scribd.com/document/984573326/3DINO-ViT)  
47. MedSegBench: A comprehensive benchmark for medical image segmentation in diverse data modalities \- ResearchGate, accessed February 1, 2026, [https://www.researchgate.net/publication/386105712\_MedSegBench\_A\_comprehensive\_benchmark\_for\_medical\_image\_segmentation\_in\_diverse\_data\_modalities](https://www.researchgate.net/publication/386105712_MedSegBench_A_comprehensive_benchmark_for_medical_image_segmentation_in_diverse_data_modalities)  
48. Predicting Redox Potentials by Graph-Based Machine Learning Methods \- ChemRxiv, accessed February 1, 2026, [https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/65f80db166c138172938cd83/original/predicting-redox-potentials-by-graph-based-machine-learning-methods.pdf](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/65f80db166c138172938cd83/original/predicting-redox-potentials-by-graph-based-machine-learning-methods.pdf)  
49. Deep learning-based lung cancer classification of CT images \- PMC \- PubMed Central, accessed February 1, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12210548/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12210548/)  
50. Big-Data Science in Porous Materials: Materials Genomics and Machine Learning | Chemical Reviews \- ACS Publications, accessed February 1, 2026, [https://pubs.acs.org/doi/10.1021/acs.chemrev.0c00004](https://pubs.acs.org/doi/10.1021/acs.chemrev.0c00004)  
51. Tri-Reader: An Open-Access, Multi-Stage AI Pipeline for First-Pass Lung Nodule Annotation in Screening CT Technical Development \- arXiv, accessed February 1, 2026, [https://arxiv.org/pdf/2601.19380](https://arxiv.org/pdf/2601.19380)  
52. In the Picture Medical Imaging Datasets, Artifacts, and their Living Review, accessed February 1, 2026, [https://orbit.dtu.dk/files/407890109/3715275.3732035.pdf](https://orbit.dtu.dk/files/407890109/3715275.3732035.pdf)  
53. The Hallmarks of Predictive Oncology \- IDEKER LAB, accessed February 1, 2026, [https://idekerlab.ucsd.edu/wp-content/uploads/2025/01/Singhal-CancerDiscovery-2025.pdf](https://idekerlab.ucsd.edu/wp-content/uploads/2025/01/Singhal-CancerDiscovery-2025.pdf)  
54. The Hallmarks of Predictive Oncology \- PMC \- PubMed Central \- NIH, accessed February 1, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11969157/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11969157/)  
55. Lung Nodule Classification \- CatalyzeX, accessed February 1, 2026, [https://www.catalyzex.com/s/Lung%20Nodule%20Classification](https://www.catalyzex.com/s/Lung%20Nodule%20Classification)  
56. A comparison of the fusion model of deep learning neural networks with human observation for lung nodule detection and classification \- PubMed Central, accessed February 1, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8248221/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8248221/)  
57. A Lung CT Foundation Model Facilitating Disease Diagnosis and Medical Imaging, accessed February 1, 2026, [https://www.medrxiv.org/content/10.1101/2025.01.13.25320295v1.full-text](https://www.medrxiv.org/content/10.1101/2025.01.13.25320295v1.full-text)  
58. MedGemma 1 model card | Health AI Developer Foundations, accessed February 1, 2026, [https://developers.google.com/health-ai-developer-foundations/medgemma/model-card-v1](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card-v1)  
59. Generative AI for Healthcare: Fundamentals, Challenges, and Perspectives \- arXiv, accessed February 1, 2026, [https://arxiv.org/html/2510.24551v1](https://arxiv.org/html/2510.24551v1)  
60. Daily Papers \- Hugging Face, accessed February 1, 2026, [https://huggingface.co/papers?q=3D-to-4D%20domain](https://huggingface.co/papers?q=3D-to-4D+domain)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAVCAYAAAA98QxkAAACC0lEQVR4Xu2WT0hVURDGv0qCKAgMbKFJlBiFUChEhQiRLdwk2MIgaOWqsk0u+iMIUmCI2X+LIItAEBctFCKigqxWEUWCLYR2gW0KXUhB2Pc5I284IaTx6Arvgx/vzMw59847d+aeCxRUUEH/VfvIU/KVzJIZ8pbci5OyqEewhGvSQBa1ErbDn9JAVrUHtrvX00BW1Q5LuCkNZFUvyU+yPg1kUUryFyzpv9UG0kWGSQ+5SnaSzjgpX1IZqBwWutlpsjHY+8kEORZ8+tOTpDX48qYbsIQPpAGqmLwO9i7yndQH37wekqrUmQ+Nkx9kbeJfQR6QE8H3nNwPdtQh/60kN2F/oJsMkiMeW0MukcvkInkDu08FuQ07rE6SNthhpkPt3NxK1zbY7ioYpQtfI9PINeJ2LPwkos6QctgmbCUvSIvH+mD1Luk6H3ysBLVGp+xu92neCFktow6W5HtYEl/cfgd75PKJAU12aQflWxd8teQCbG0/2Uw2kWb3RZXCmlu7KZ0it3ysNYfJY7elV6Qx2IuWjmwlnL76tri/LPjUFx3BlhrI52APkaPINbRK5ayPtatTsB5asnR8P4G9xual+pM9FnzSR3Iw8VUj9+osgT3VHaTXfarnvT7Wkxv18T9J5XAFVjrPyB3YTWNjriLf8GcTS9pFNdF5WOPdhX0WaI2+ZYp83nFYUy5P/QZJUGNJNWL5iQAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAAoCAYAAABDw6Z2AAAFNklEQVR4Xu3de6i0UxTH8eV+v0Qoktc1Sm6vOym3EKH4g5Lyj5JLkRTidS13iUQRUkpu4Q9JhKIIKXIJeROKP9z+ICTWr/1sZz3rzHPOzLxnzsw58/3Uavbs59lzzjPn1KzW3s8eMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlpTLPK7pESfEk5a4fG2K81tndHvI422P3fMBAACAcbg+d8COaR4/91g/HgAAAMufqjzVvqE9TtOcsD3ZPK7d6p3xr8fOuRMAACxv14b2IaE9TtOcsL3YPK7X6p3xWO4AAACzPWelyrFcxITt0NCOdvX4wmaqPtt7fBuez0fj9Z71O36xErZPrf23fMfjBo/NQl+/Hrb2a93l8azHitDXj5qwbdjqNdvf406PXTx2SscAAEDyo8dnuXMJuy60uxI2UTJyuscGHs/b4NU4JS8aL/ONX6yE7VJrJ1kXhPagVlr7tR73WCc871dXwqbXO9rjZmNKFACAeelD+d7cuYTF5Gi+hO01j6c9jkjH+qFF8xq/sc0//srcMSJbevzaPN6Yjg3jVY9jPU6x4ZI16UrY9P4r/rDhXxsAgKnxl8cWuXMJi4nKYaGdXWxrPhWs8ZpSniT7ePxkC5MEad2ZrvHAfGAAXQkbAADokxK1N3NnsrWVD12tN7rHypqoSXZTaM9V+XrG4zePrfKBPmkrCo3/Mx8YsyusJFkn5wONvaws9L/dY5XHUe3DLaoi6rVuzQcGQMIGAMAaus/KlFf0Vmj/4nFceC57p+cL7UybmS7ris3/P3u2WGE7PLQrrVl7qmnv6fGRdd8s0Esdf4aV8fp9Bhk/Skqq1/JYbb0TSW1qu0Pq+yA9r5Ro6WYFrTXrqkQqmb/FyrlK5ntto9KVsOk9e9njE49XPN6w8nN2jCcBAIByV+Em4bk+7C8Mzx8N7WgPKx/kqtKc1fRtZOVOQi0i1+vs5vGAx0Uel1tJnq7yeKI5f1RiBTBPiWqKT1OYJ4Y+JQn1GipVz672uM3KtUR5vJKOPH4c9N5W+t1zkqV1bZoqzU7NHVYqa/p7iipwXUnplx7nNm1Va38Ix6quhE3Ul9/Lru0/AACYOqpofGjlg1htVVlq9arSB/mmTftIK1ONOneFlSqI2pWqNv807UuaR52jKlR1WvO4OvSNQrzpoN65qarde1auT0mFrkeUTNbr1hYY1Tkev3scFPo0Xuv98vifbfZ4UdKkLUZywreQ9LVOmpbVz/879H/V9OlbBPR3UvJ1h5Wvgqp0o4mOKcmudG35/6C+b7qbWOfXJFg3XGS9kr+5Era6BUtNDrepBwAAQH9WWvtmBO2XFT/IV4X2SR5fN21NF27XtOvaJ1Ws6loxTcNq37JRiQnbwaE9CCWgZ9vsJGwQtVI57Bq5haYtP+L1nGfDb+eidXDxf2EucyVsL1lJGN/PBwAAQH80BaY1UaIqkdofzxy240P7ACs3L2zr8b3H3U1/rejUipRoalTVnVFVnmLCFitkg9C+alLXug1DFTp9wfmkUBKtzX2r1z3uD88HoUpaTNj091X17ZHQV3UlbJr6VHVQFmvbEwAAMCHid4nuF9qLqVYfh00YJ129YaBS8h2T+eiF5nHdVm9Z1/hu6gMAAFNCU5nVuO441Ia9qvSN+gaLcdL6Rq191Oa6D1r7ZpVIa/kyTZF/4/GdsW4NAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABMqv8AC1La+2oKUxYAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABcAAAAYCAYAAAARfGZ1AAABL0lEQVR4Xu2UvyuFURjHv+RHGdU1MBh0yz8g6pqUMgoDJXex4S9QZL0lk4mMFr8yWhSbIsPtTtdoMSlFsfA5PefVe87EPTeD7qc+9Z7ne3rO+573vK/U4l8ziFf4ip/eOyz6fC9Xf/ZzO3z2Y4ZkDerYnqv34yMuRfVfcypbYNqPu/EcR79nJDAha36JPXiMpWBGIjXZAtc4GWXJrMman8VBKl14gi/4jr1h3DhZ4zLuyO5+NZjRIO5UHOGsHw/LmleVePw6Zfs7FdUvZAvMR/UM96TruIEVbAtjqU/WeDsOYEbW/D4OPMu44q8flHs/c3iLH7IGTziehbK9dp969tnf4G4udyziG+7jSJQl4bZyABfwQLZ409iU/RYc7q7dYWgaY7KXuYWHWAjjFn/JF4DkOrtWcP2XAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAYCAYAAADpnJ2CAAABaUlEQVR4Xu2VPSiFURjHn+SjTDJQDJIUk5Eig1Im+RqUIgYWBpPRZvE1iMVkU76yKClhUMrHIKVsslgoZWDhd3rO1bnPvbfkvu+i91e/4fyf857n3nvOe65IQkKe1OApvuOX9wrrfX09yF/93EJfy4s60UUfsCDIq/AJh00eCbuiTXv8uAT3sflnRsR0iDY8xlLcxta0GTFwJ9r0DDtNLRamRBvu2UIcFOMOvuEHlqeXs9KIG7iJ8ziL7TgaTspGqtkILot+y8m0GZnM4C22BJnbc/dsU5Bl4E7jFvb7cYPoQ26xXK/COD5itS3AtQ1CikT3q8vkh6JNB03uKMMXHLMFT7cNUlSINlu0BegTbXhjCzCBn6Kvzq8YwEvRh9yiz9gW1N3euWssdaVd4GpQX8DzYOwYwhU8wiXJvRV/Ylr0Q1jcyby3YRRUit6vvUHmft4TXAuySKkVvfrcP8sBzom+HvbwJfwDvgF4/UbCewMj8gAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAVCAYAAAAAY20CAAACTElEQVR4Xu2WTYiOURTH/z5qyMwIxcjGzMZCyjcpk1A0SpKiZsrKBqGRhbIi8rHxuVU22AyaZjcLX42ZCTOzMiiWFpTEQmji/++cZ+Y48wpZPXp/9et97rn3fZ773HvOfV+gSpUq/xVraDd9S7/Tz/QJvRoHlYHbsBdYljvKwETYDjzPHWVhNWz1L+WOsnAM9gLbc0dZeEC/0um5owxo0iOwl/gTttBB+gV2eslntIc2hnFiE31Iv9EbdFXoO0dfwXZ+nscmwe6tmOpxhsdn04N0Pz3psVGUNvrC8dzhHKZzUuwIvZdiu+lTOiHFj9JHKVZwgg7BxhTU0muwg0XsoXcxfg6jXIa9wIbcQWbCVjbTQU+lmB6oNIyrLLRDZ1NMaLxqby997W2xke7066X0HZ3v7YoMw9JhWoprJbUS+1JcD9JNlR6RKbBU3BZik2H3VtplltOtsBT+SFs8rpdq8OteesavK7IAtvpapchUepF+wvjCXgybqLY6shJ2r0Uhpt2oNFYcorP8+gq96dcX/FP1pPut8PZPNMMmrfzToDfeHqAfPCavF18IqCYqpZXyWAUYaad9KVYQa24hbKdUzOc9tgM2h7yA/8wtejrF6uhLui7F9fdEO5lRGuZD4z7sYNjl7bWwF9AJFKlP7b+iyP/NIaY06MLYgws09j3GCjKiNGxNsTbYhOd6WzWpRVGqFehZvzrRfst62gl7iPJUJ4vSTvnbFMYJFfId2FidWPEP4gH6gj6mS0K8BrYLEaWWfmO0QDpQ+mGFX25+AC+mdTa7l4YuAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACsAAAAVCAYAAADfLRcdAAACQElEQVR4Xu2Wy6vNURTHF8ojr7jyKIQBA3HzTF7lUSYykH/AgBTiEibII2TiPSORYuQxoJQi8swjSUgGUggDZCBKfL5n7d/9rbO7xZnoNzif+tRda+3fOfu39177XLMmTZr8N6bjFfyIv/E7PsDjcVDVOG8+2Ul5oWp0Nl/ZF3mhikwzX9XDeaGKbDaf7OK8UEVu4E/smxeqhib4y3zC/0Ib7g/xEDxovjPHcBtewOc4sxxWYyCuwVW4Cy/hrFTrgm/wrvnt9DbFdWjr9UU78kJiPQ4K8WScGmKhG0Q70yPk1uKrEC/Da1b/WaPxMw7GObgh1G7huhDXOGI+2Xl5AfqbP/Q3tFL5zqgPnqW/J+InHNFeLXmIy3Erdk05vbRefkIxqEDb9QN7ZvlOeBJXprg37sSrOKAYlDiDe0KsyV3H8Sm+g3vLch1PcRMOD7n55iuuK7WdMearqjMS0Zsdwm9WNt0K7I6vcVTKFbzDU7jFfNJPrFzFkebfMSXFEW1/R7eQFkXnucZs8wk+Nh+sL1P8CL+knDxdPADDzJ/TT3Gk1XysjkzBQvMXHYpLUr2jm0ZHRY3UK8tr9zZmuYbRj4YaThMvWG2+khGdNU1Q/3eo2/W3boJIH3yPS7O8dk/HckaWbwhdLR/Mu/lAyOu87guxOIH3zRtGvfDS/HYoaMHbuDvkCuaaN1e3vNAIajhNQBMdZ94QR/ErnsPteBYvp3w/f6zGWPOb4aJ5097DRaEudP3pzN80P0L6jAV1I6rOH7Tqc6yN0TVYAAAAAElFTkSuQmCC>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACsAAAAVCAYAAADfLRcdAAACIElEQVR4Xu2WP0iVYRTGj2mQWAQVuQQOEqUOISJkoIuog4vWECQ4tIUNRktmOEgKRamoOfoHwUEEhRLFFhF0SUKXcnKoaKhBsSFKQp+ncz7v69Hh0yG+4D7wg/d93u/ee+75894rklZaaf0z3QBvwTewA36CZTAYPpQ0TYoGW+IPkqYTopld8wdJ1HXRrPb5gyTqiWiwN/1BErUAfoOz/iBpYoB/RAOOo3qwCrbBnGgLUU2i1fkKqsw7TPfABqj1B3HE0vND2v2B6SHIdR7bZinYnwR3QSnICPzDlCUa7Gl/EEf9osFW+gPoHFj0JjQNntn6MuiR+C1UBt57M64+gl8gx/nM0IhoeUMxM5uiZawDb8zzKgAT4CWYAsXmt4BX0UOm++bxNmowj9fp470noCuiWeUvWKhs0At+yMGMsUfZr49E3/yTaBuEugS+gCLbs89f23oG3LY1xfeJWrAQvLB1BxjlokI0wBVJDQX3LA+zRo+M/X3ZfjEzn8FVcEr0x6Rx3xPaIuFro6FkBbbAefOZCCbkOXgqWkV6Z0Rvp+jLHlvMTFewZ5nDYaN4s3DgIjEYPlcO3gU+/5cwUV41YN2bR1Wm6CTfCrxrolXgTRBpCNyx9QXR4Jkt3izd4AG4CPLAd0n1PPu6GeSDD+YdS7w7OUwMbFxSJWLW6LGlqs3jh82CNtHh4rBRvHHmQavtKQ7UMBgAnZIadPYsb5n/R7tI8mnNHfxr+wAAAABJRU5ErkJggg==>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADcAAAAUCAYAAAAp46XeAAABhUlEQVR4Xu2WPUvDUBiFX79wEUXQxUV0FN1cBBcdHJzERdDNwUl08AcIOrk46eAmiCIi+AUu4iZYERc3B3+Bg5MIgug5vAlNX5qbXklKI3nggebctM25yb2tSEFBLmiCAzaMYQPu27AR6YMz8AaembFqcAI+4IEdaDRW4DPcgV9SWzme8y05KBeFdyOp3BTclZzcuShJ5VrgPewVv3Kzous5jmnYZsO0SSq3BNeC1z7lluGmDQP4eXviLp8KrnKd8EHKM+xTjmzDRZNNwivRJyJzXOW2RHfUEN9yzfAILgTHY6K7MyetLvCCz20I+uG1yXzLkVZ4DNfhLeyuHM6WuHIncMhkfylHJuAbXLUDWcMLvrChaP7j0K6lOMbhI+yBh3C+cjhbWOLShlXgxbGUz53jGnuBg8ExNxG+vy4F2+Gn6FpI2pb5d43luEHUwqhosWGTcw3ySZkzeWpwOy6JFgsfs1fRnawjcl7IKXyX8rlPoj/CLjhhIzYM4HfcwS47UFDwz/gF6G1U0SvtthwAAAAASUVORK5CYII=>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFQAAAAUCAYAAAATMxqtAAAB4UlEQVR4Xu2XTShlYRjHH18pn42kRinsFDsbNRuWomSmpqjxsdAUsVCys7BTrKSsFPkoygzLaSxkoWYzzYIk2dhRakrKhv/fc+7tnMd9Tzcr93h/9Vuc53nr1v+8532fK+LxeN45ebDBFkMUwyl4AH/BOVgQWeF5phb2iIa0a3opykWDXBUNsQaew+/hRR6RcfgPLsIHcQfK3XgjuktJF3yEy+kVnhfcSeZAq+A9XArVSkVfQnOo5jG4Ah0Q3Y3cza+hV/R8dtEJi2wxCbgCnRcNdAT+hIdwB7aGF8UwBmdtMWBS9NiICzxncQW6IhroGWwKat/gf8n+k1+Aw6bWAfclwZOCK9BN0UAZSgruqGu4HarFkQ83YH/w3CY6VVSkVyQQBvrDFkUvIwZqd9iJ6GRQZuouCuEWnIG/4YdoO3m4AuX5x0C7TZ3jFusfTT2OdtGdPWEbSYSB8tKxfBYN7oupnwZ1jlDZ8An+gdVwHfZF28mDge7ZIiiBt3A6VOMZykGf/56ygWcmL7XG4JkX0ZokOFT+A+LwzrMt0wgzBC9hXfA8CK9gS2pBDByvGKadCHim8ov4auo5DUeXY9Ew+fnSC9Eb2F42o0Hvr2i/PtJ1w5fkCp6/cQQrbcPj8XjeME8JcFsYyq/ZxQAAAABJRU5ErkJggg==>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADcAAAAUCAYAAAAp46XeAAAB0klEQVR4Xu2WTShEURiGP38pv5EUpbBT7GyUDUtRQinK30KKWCjZWdgpVlJWivwU5W8pFrJQNrIgSTZ2lFJSNrxv353pztec26SZaWieehb3O6dm3nPu+c4VSZPmT5ABa2zRRy6cgWfwBC7ArIgZKUgl7BT9w/tmLEShaKh10UDl8AGO+SelGpPwBi7DL3GH4y69iu4eaYffcDU8I8X5kOjhSuEnXPHV8kUXpN5XS2lc4QZFd4m7/Bu6RM+zizaYY4vxxhVuUTTcKDyE53APNvonBTAB523RY1r01Q4KHxdc4dZEw93DOq82AN8l9tdyCY6YWis8liR1XFe4bdFw/IMhuNIvcNdXCyITbsF+77lJtDsXhWckGIY7sEXRRsJwduVvRTtsgam7yIY7cA6ewpLI4cTiCsfzwnAdps4rhPUKUw+iRXTHp+xAomE4NgxLt2iIHlO/8+q8FmKhGV7BMrgJ+yKHEwvDHdkiyINvcNZX45njpc6vlljgGWNDqvWe2UQ2JEkB+eXBi5pnIVpbHoZPsMp7HoLPsCE0IQBeGQxmOyvPIN+UXlOPG2zHl6LB+IrRR9FOZhvFuDd2LTpeHTHqhgvmWgT+xgUstgNp0vwzfgD+flsYgGHayAAAAABJRU5ErkJggg==>