# 1. Foundational Knowledge of OCR (Optical Character Recognition)

# Foundational Knowledge of OCR (Optical Character Recognition) — A Review Document for Researchers Getting Started

## Introduction

OCR (Optical Character Recognition) is one of the oldest fields — and still one of the "hottest" — in computer vision and natural language processing (NLP). Before starting a scientific research project on OCR, a newcomer needs to grasp the overall picture: what OCR is, what sub-problems it comprises, how the processing pipeline works, how successive generations of methods have developed, and how to scientifically evaluate an OCR system. This document compiles that foundational knowledge in a logical sequence, from definitions to the most modern models.

## 1. Definition of OCR and Related Problems

**OCR (Optical Character Recognition)** is the process of converting images containing text (photographs, scans, natural scene images, etc.) into machine-encoded text that a computer can read, edit, and search. In essence, OCR is a mapping problem from pixel space to character/word space.

However, "OCR" in modern research is usually decomposed into several sub-problems, each with its own characteristics and methods:

- **Text detection**: identifies the location (bounding box, polygon, or mask) of text-containing regions in an image, without yet reading the content. The output is typically the coordinates of a box surrounding a line of text or a word.
- **Text recognition**: given a cropped image region (already known to contain text), the model reads and decodes it into a string of text characters.
- **End-to-end OCR**: combines detection and recognition into a single pipeline (or a single model) to go directly from the raw image to text, without requiring a manual intermediate step.
- **Scene text recognition (STR)**: the branch concerned with recognizing text appearing in natural scene photographs (signs, product labels, street images, etc.) — much harder than text printed on a white background because the text can be tilted, curved, blurred, occluded, in decorative fonts, or under uneven lighting.
- **Document layout analysis (DLA)**: analyzes the layout structure of a document page — determining what is a heading, paragraph, table, image, newspaper column, footer/header — before or in parallel with reading the text.
- **Handwriting recognition (HTR/ICR)**: Handwritten Text Recognition (HTR) and Intelligent Character Recognition (ICR) are handwriting recognition problems — harder than printed text because strokes vary enormously between writers, writing speed, and script style (discrete letters versus cursive).
- **Key information extraction (KIE)**: once the text (and its location) is available, KIE extracts meaningful information fields (for example: names, dates, amounts, tax codes on an invoice) — this is the step that turns "being able to read the text" into "understanding the content."
- **Document understanding / Document AI**: a broader concept, encompassing the entire structured-document processing chain — from OCR, layout analysis, and KIE, to document question answering (Document VQA), document classification, and summarization.

Clearly separating these problems is important when writing a research proposal, because each problem has its own datasets, evaluation metrics, and baselines.

## 2. The Classical OCR Pipeline

A traditional OCR system (predating or running alongside deep learning) is typically divided into the following sequential stages:

**(1) Image acquisition:** the input image can come from a scanner, a phone camera, or a natural scene photograph. Source image quality directly affects the entire downstream pipeline.

**(2) Pre-processing:** the goal is to normalize the image so that subsequent steps operate more reliably, including:
- *Binarization*: converts a grayscale/color image into a binary (black-and-white) image to separate text from the background, commonly using algorithms such as Otsu's method or adaptive thresholding.
- *Deskew*: rotates the image to correct the tilt angle of the text (caused by misaligned scanning or an uneven shot).
- *Denoise*: removes noise (salt-and-pepper noise, smudges, ink bleed) to clean up the text signal.
- *Contrast enhancement*: increases the contrast between text and background, especially important for underexposed or color-cast images.

**(3) Layout segmentation:** divides the page image into text blocks, then further separates them into lines (line segmentation), words (word segmentation), and finally characters (character segmentation) — this is a highly error-prone step in the classical pipeline, especially for handwriting or connected characters.

**(4) Feature extraction:** for each segmented character/word, the system extracts features describing its shape — for example: histogram of oriented gradients (HOG), projection profiles, the number of stroke crossing points, bounding-box ratio, and geometric/topological features (number of holes, number of stroke endpoints).

**(5) Classification/recognition:** uses a classifier to map the feature vector to a specific character label (e.g., 'A', 'b', '7', etc.).

**(6) Post-processing:** uses a language model or dictionary to correct spelling errors, remove nonsensical recognition results, and improve the overall accuracy of the output text string — for example, correcting "rnodel" to "model" based on n-gram probabilities or a valid-word dictionary.

This classical pipeline has the advantage of being transparent and easy to debug step by step, but its major drawback is error accumulation across stages (error propagation) — if character segmentation is wrong, the recognition step will almost certainly be wrong as a result.

## 3. Traditional OCR Methods (Pre-Deep-Learning)

Before deep learning became widespread, character recognition relied mainly on classical machine learning techniques:

- **Template matching**: directly matches the character image to be recognized against a predefined set of "templates," computing a similarity score (typically correlation or pixel distance) and selecting the closest template. This method is simple and effective for a fixed font (e.g., printed text in a single font) but is very sensitive to distortion, rotation, and scale, and does not scale well to multiple fonts or handwriting.
- **Feature-based classification**: instead of matching raw pixels, invariant features are extracted and then classified using classifiers such as k-Nearest Neighbors (k-NN), Decision Trees, or shallow Neural Networks.
- **Hidden Markov Model (HMM)**: models a sequence of characters as a hidden Markov process, well suited to sequential problems such as cursive handwriting recognition or word recognition without requiring explicit prior character segmentation — an important advance because it reduces the dependence on perfect segmentation.
- **Support Vector Machine (SVM)**: widely used as a character classifier due to its strong separation ability in high-dimensional feature spaces, particularly effective when combined with HOG features or Gabor filters.

Overall, traditional methods require very labor-intensive hand-crafted feature design, and their performance depends heavily on the quality of the pre-processing/segmentation steps.

## 4. The Deep Learning Revolution in OCR

Deep learning has completely transformed the approach to OCR by enabling features to be learned directly from data (representation learning) instead of being hand-designed, while also solving the core problem: *segmentation-free recognition*.

- **CNN (Convolutional Neural Network) as a feature extractor**: a CNN extracts a feature map from the image, capturing local patterns (strokes, edges, corners) hierarchically — completely replacing hand-crafted features such as HOG.
- **RNN/LSTM/GRU for sequence modeling**: since text is inherently a sequence of characters, recurrent neural networks such as LSTM (Long Short-Term Memory) or GRU (Gated Recurrent Unit) are used to model contextual dependencies between adjacent characters within a line image.
- **CTC loss (Connectionist Temporal Classification)** — this is the single most important turning point: CTC allows a sequence-to-sequence model to be trained **without knowing the alignment in advance** (i.e., without knowing exactly which character corresponds to which pixel position in the image). CTC automatically sums the probability over every valid alignment between the input feature sequence and the output label sequence, using a special "blank" symbol to handle repetitions/non-emitting positions. As a result, the model can learn directly from (line image, text string) pairs without needing character-level labels — greatly reducing data-labeling costs.
- **CRNN architecture (Convolutional Recurrent Neural Network — Shi et al., 2016)**: the classic architecture combining a CNN (feature extraction) + bidirectional RNN (BiLSTM, context modeling) + CTC loss (sequence decoding). CRNN became the standard baseline for text recognition for many years due to its simplicity, effectiveness, and lack of a need for character segmentation.
- **Attention-based encoder-decoder**: instead of CTC, another approach uses an attention mechanism so that the decoder "looks" at different parts of the image while generating each output character in sequence — similar to neural machine translation models. This approach usually handles curved, rotated text, or complex reading orders better.
- **Architectures based on Transformer and Vision Transformer (ViT)**: recent models replace RNNs with self-attention (Transformer) to model global relationships between sequence elements, parallelizing better and training faster; ViT splits the image into patches and applies the Transformer directly to the patch sequence, extracting image features without convolution.

## 5. Modern Text Detection

Deep learning methods for text detection focus on solving the problem of detecting text regions with diverse shapes (horizontal, tilted, curved):

- **EAST (Efficient and Accurate Scene Text detector)**: a direct-prediction (single-shot) model that predicts which pixels belong to a text region and directly regresses the bounding-box geometry (rotation angle or quadrilateral) for each pixel, skipping intermediate steps such as region proposal — well known for its speed and good accuracy.
- **CRAFT (Character Region Awareness For Text detection)**: instead of directly predicting a word/line box, CRAFT predicts a character-level region score map and an affinity score between characters, then groups them together — this approach handles arbitrarily shaped text (curved, distorted) very well because it does not depend on fixed box geometry.
- **DBNet (Differentiable Binarization Network)**: improves the binarization step (which is inherently non-differentiable and hard to optimize end-to-end) with a differentiable approximation, allowing an adaptive threshold to be learned during network training — significantly speeding up inference while retaining high accuracy, and becoming one of the most popular baselines today. Later variants (DBNet++) further improve multi-scale feature representation.

## 6. Modern Text Recognition

Once a text-containing image region is available, modern recognition models have gone well beyond basic CRNN:

- **ASTER**: combines a rectification module (based on a Spatial Transformer Network) to "flatten" curved/tilted text regions before feeding them into an attention-based encoder-decoder, handling scene text with unusual shapes well.
- **SATRN (Self-Attention Text Recognition Network)**: completely replaces RNNs with two-dimensional self-attention, modeling dependencies along both the horizontal and vertical directions in the image — useful for text with complex layouts, strong rotation, or heavy curvature.
- **SVTR (Scene Text Recognition with a Single Visual Model)**: a "visual-only" architecture that requires no separate language module, exploiting local/global feature-mixing blocks inspired by Vision Transformer, achieving very fast inference while remaining competitive in accuracy — suitable for real-world deployment.
- **PARSeq (Permuted Autoregressive Sequence model)**: trains the model on multiple different sequence-generation orders (permutations) within the same architecture, allowing it to decode either autoregressively (sequentially, leveraging already-generated context) or in parallel (non-autoregressively) when speed is needed — one of the state-of-the-art models on several recent scene text benchmarks.

The general trend among modern recognition models is to rely increasingly on attention/Transformer mechanisms, reduce dependence on sequential RNNs (which are hard to parallelize), and integrate linguistic context directly into the visual architecture rather than separating it out as a distinct post-processing step.

## 7. Document AI / Understanding Structured Documents

As the problem expands from "reading text" to "understanding documents" (document understanding), a distinct research direction has emerged:

- **The LayoutLM family (LayoutLM, LayoutLMv2, LayoutLMv3)**: extends BERT-style language models by adding 2D spatial position embeddings and image features for each text token, allowing the model to jointly learn textual content, on-page position, and surrounding imagery — very powerful for tasks such as KIE and document classification.
- **Donut (Document understanding transformer)** and **Pix2Struct**: represent the **"OCR-free"** direction — the model reads the document image directly using an image-to-text encoder-decoder architecture, producing an output sequence (which can be structured JSON) **without needing a separate OCR engine** as a pre-processing step.
- **TrOCR (Transformer-based OCR)**: uses an image encoder (an image Transformer, e.g., pretrained ViT/BEiT) combined with a text decoder (a text Transformer, e.g., pretrained RoBERTa/GPT), leveraging the strength of pretrained models on both sides to recognize text, particularly effective for both printed and handwritten text.
- **Why the "OCR-free" direction is becoming a trend**: the traditional OCR pipeline (detect → crop → recognize → parse layout → extract field) has many discrete steps, each a potential source of error (error propagation), is hard to optimize end-to-end, and requires labeling costs at multiple levels (box, character, information field). OCR-free models treat the entire problem as a single mapping from image to the desired output sequence, learned end-to-end with a single objective, simplifying the deployment pipeline and typically generalizing better to novel document formats.

## 8. Multimodal Large Language Models and New Dedicated OCR Models

The explosive growth of multimodal Large Language Models has opened up a completely new approach to OCR: using a general-purpose foundation model to "read" images as a secondary capability within a broader skill set, rather than training a dedicated OCR model.

- **GPT-4V, Gemini, Qwen-VL**: these general-purpose vision-language models can read text in images quite well thanks to training on a massive volume of image-text data, while also being able to reason over the content read — for example, answering a question about a chart or summarizing the content of a document page — something traditional OCR cannot do.
- **GOT-OCR2.0**: a new-generation dedicated OCR model ("OCR-2.0"), designed to handle a diverse range of input types (plain text, math formulas, tables, musical notation, charts, etc.) within a unified architecture, aiming to become a next-generation "general-purpose OCR engine."
- **Kosmos-2.5**: a Microsoft multimodal model designed specifically for text-image-related tasks, capable of generating text along with spatial information (layout-structured markdown) directly from a document image.
- **Nougat (Neural Optical Understanding for Academic Documents)**: an OCR-free model dedicated to scientific documents, trained to convert PDF page images of scientific papers into markup-formatted text (including LaTeX math formulas and tables) — addressing a major weakness of traditional OCR, which does not handle complex mathematical formulas well.

A point worth noting for research: general-purpose multimodal LLMs are usually strong at semantic reasoning but may be less accurate than dedicated OCR models at the character level (character-level fidelity), especially for dense text, numerical data, or less common languages — this is a very worthwhile empirical evaluation direction for a research project.

## 9. Evaluation Metrics

Evaluating an OCR system requires separating out each sub-problem:

**For text recognition:**
- **CER (Character Error Rate)**: the character-level error rate, computed as `CER = (S + D + I) / N`, where S (substitution — number of incorrectly substituted characters), D (deletion — number of missing characters), and I (insertion — number of extra characters) are computed from the **edit distance** (Levenshtein distance) between the predicted string and the ground-truth string, and N is the total number of characters in the ground truth. Lower CER is better.
- **WER (Word Error Rate)**: similar to CER but computed at the word level instead of the character level — usually stricter than CER because a single wrong character in a word causes the whole word to be counted as wrong.
- **Edit distance**: the mathematical foundation for both CER and WER, measuring the minimum number of operations (insertion/deletion/substitution) needed to transform string A into string B.

**For text detection:**
- **Precision**: the proportion of predicted regions that are correctly text, out of all predicted regions.
- **Recall**: the proportion of actual text regions correctly detected, out of all text regions in the ground truth.
- **F1-score**: the harmonic mean of precision and recall, commonly used as the single combined metric for comparing detection methods — a predicted region is usually counted as "correct" (a true positive) when its overlap (IoU — Intersection over Union) with the ground truth exceeds a predefined threshold (e.g., 0.5).

**General interpretation**: no single metric is "sufficient" on its own — a system with high Recall but low Precision means it detects a lot of real text but also falsely flags many non-text regions; low CER but high WER means errors are small at the character level but spread across many words. When writing a research report, multiple metrics should be reported together, with the measurement conditions clarified (e.g., the IoU threshold, whether case is normalized, whether punctuation is stripped when computing CER), because these measurement conventions vary considerably between papers.

## 10. Common Datasets/Benchmarks

Choosing the right benchmark is an important factor in positioning a research project relative to the state of the art:

- **ICDAR series** (ICDAR 2013, 2015, 2017, 2019, etc.): the oldest and most influential series of competitions and benchmark datasets in the OCR community, covering text detection, recognition, and many other sub-problems over the years (ICDAR2015 is well known for tilted/incidental scene text).
- **IIIT5K**: a scene text recognition dataset with 5,000 word images, collected from Google Image Search, a standard benchmark for the cropped word recognition task.
- **SVT (Street View Text)**: text images collected from Google Street View, characterized by low image quality and complex background noise.
- **COCO-Text**: a large-scale natural scene text dataset, built on top of MS COCO images, used for both detection and recognition.
- **SROIE**: a benchmark for extracting information from scanned retail receipts, a popular standard for the KIE task.
- **FUNSD**: a dataset for the form understanding task, labeling relationships between fields (e.g., "key-value" pairs) on form-type documents.
- **CORD**: an Indonesian receipt dataset with detailed hierarchical structure labeling, commonly used for KIE on semi-structured documents.
- **DocVQA**: a benchmark for question answering directly on document images (Document Visual Question Answering), requiring a model to both read the text and reason about position/context to answer — well suited for evaluating modern document understanding models.
- **Newer benchmarks**: also worth mentioning are **Union14M** (a large-scale scene text recognition dataset aggregated from diverse sources, aimed at addressing the fact that older benchmarks have nearly saturated in accuracy), **HierText** (a benchmark with hierarchical line/paragraph/block layout labels), and various multi-page, multilingual document benchmarks aimed at evaluating the capability of OCR-free/multimodal LLM models on entire complex document pages rather than just individual words.

## Conclusion

The overview above shows that OCR has gone through a clear developmental journey: from classical statistical/geometric methods based on hand-crafted features, through the era of CRNN and CTC loss solving the alignment problem, to modern attention/Transformer architectures, and most recently the trend of unifying the entire pipeline into OCR-free models or leveraging the existing read-and-understand capabilities of general-purpose multimodal models. For a research project just getting started, mastering this conceptual map — knowing which sub-problem needs to be solved, which method is appropriate, and which benchmark/metric to use for measurement — is an important preparatory step before moving on to experimental design or proposing new methods.
