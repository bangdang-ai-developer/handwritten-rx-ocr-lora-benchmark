# History of the Development of Optical Character Recognition (OCR) Technology

## Introduction

OCR (Optical Character Recognition) is the technology that allows machines to "read" text from images or paper documents, converting it into digital text that a computer can process, search, and edit. From rudimentary ideas conceived more than a century ago, OCR has gone through many stages of development: from crude electromechanical devices, through the era of pattern matching and statistical recognition, to the deep learning revolution, and today, massive multimodal language models. This document reviews that journey along a timeline.

## Late 19th – Early 20th Century: The Founding Ideas

The idea of a "reading machine to help the visually impaired" appeared very early, driven more by humanitarian motives than by a data-processing goal. In 1913–1914, Irish physicist **Edmund Fournier d'Albe** invented the **optophone** — a device that used a photosensor to scan across each printed character and convert the letter shapes into sounds of different pitches so that blind users could "hear" the letters. This is regarded as one of the first attempts to turn the optical signal of writing into information that a human (or, later, a machine) could interpret — a conceptual precursor to modern OCR, even though it did not itself "recognize" characters in the fully automated sense.

Around the same period, other inventors such as **Emanuel Goldberg** in Germany were also researching systems for scanning and searching for characters on microfilm in the 1920s, laying the foundation for the idea of machine-based text storage and retrieval.

## The 1920s–1930s: Gustav Tauschek and the "Lesemaschine"

The next important advance came from Austrian engineer **Gustav Tauschek**. In 1929, he filed a patent for a device called the **"Lesemaschine"** (German for "reading machine"). This device used optical templates overlaid on the image of the character to be recognized: if the light passing through matched a specific character template, the machine would "recognize" which character it was. This is precisely the **template matching** technique — one of the two foundational principles of OCR (alongside feature extraction, which emerged later). Tauschek went on to file several related patents in the United States in the early 1930s, extending the idea to both numeral and printed-character recognition.

## The 1950s: The First Commercial OCR

The post-World War II period saw OCR move from the laboratory into the marketplace. In 1951, American engineer **David Shepard** built **"Gismo,"** one of the first character-reading machines capable of converting printed text into telegraph code that a teletypewriter could reproduce. That same year, Shepard founded the company **Intelligent Machines Research Corporation (IMRC)** — considered the world's first commercial OCR company. IMRC sold document-reading systems to major clients such as Reader's Digest (for reading sales figures) and the oil company Standard Oil. In 1954, IMRC was acquired by Farrington Manufacturing (later associated with the credit card and retail industries), bringing the technology into broader commercial use.

## The 1960s–1970s: Font Standardization, Banking and Postal Applications, and Ray Kurzweil

This was the period when OCR "matured" industrially.

- **Standardization of machine-readable fonts**: Because early OCR machines could only accurately read a single fixed typeface, the industry developed dedicated font sets: **OCR-A** (introduced around 1968, standardized by ANSI in 1970), with simple, machine-distinguishable character shapes that looked rather "rigid" to human eyes; and **OCR-B** (designed by Adrian Frutiger, standardized by ECMA/ISO in 1973), softer and more readable for humans. These two fonts were widely used on passports, airline tickets, and invoices.
- **The banking industry**: in parallel, the banking industry developed **MICR (Magnetic Ink Character Recognition)** technology — recognition of characters printed in magnetic ink at the bottom of checks (the E-13B standard emerged in the late 1950s and became widespread throughout the 1960s–1970s). MICR is not strictly optical OCR, but it played a similar role in automating the processing of financial documents.
- **Ray Kurzweil and Omni-font OCR**: the biggest breakthrough of this period was the work of inventor **Ray Kurzweil**. Rather than reading only a few fixed fonts, he developed **"omni-font OCR"** technology — capable of recognizing printed characters in virtually any typeface. In 1976, combining this technology with a text-to-speech synthesizer developed by his company, Kurzweil launched the **Kurzweil Reading Machine** — the first machine that could read books aloud for visually impaired people, hailed as the most important invention for the blind since the Gutenberg printing press. The blind singer Stevie Wonder was among its first customers and also an enthusiastic advocate for the product.

## The 1980s–1990s: Personal Computers, Commercial Software, and the Seeds of Tesseract

The rise of **scanners for personal computers** in the 1980s brought OCR closer to everyday users, no longer confined to expensive industrial systems.

- Many commercial OCR software packages emerged to run on PCs, most notably **OmniPage** by **Caere Corporation** (launched in the late 1980s), which was the industry's gold standard for many years.
- In Europe, the company **ABBYY** was founded in 1989 in Russia (then still the Soviet Union, under a different original name before being changed to ABBYY), and later developed **FineReader** — one of the most accurate OCR software products, particularly strong with diverse languages and complex documents.
- Notably, the **Tesseract** project — which later became the world's most popular open-source OCR engine — originated in **1985 at Hewlett-Packard (HP) Labs**, was developed continuously until around 1994–1995, and was then discontinued. It lay dormant for more than a decade before HP and the University of Nevada, Las Vegas jointly released the source code publicly, and in **2005 Google took it over, continued its development, and fully open-sourced it**, turning Tesseract into the foundation for countless free OCR applications afterward.
- This period also saw the development of **ICR (Intelligent Character Recognition)** — handwriting recognition technology, much more complex than reading printed text because each person's handwriting differs. ICR was heavily applied in form processing, particularly in the postal industry (recognizing handwritten addresses to sort mail) and banking (reading handwritten amounts on checks).

## The 2000s: Statistical Methods and Early Mobile OCR

In the 2000s, recognition techniques based on **statistical models**, particularly the **Hidden Markov Model (HMM)** — which had been very successful in speech recognition — were applied to OCR, proving especially useful for cursive handwriting recognition and for the scripts of many languages without clear character boundaries. This was also the early period of **mobile OCR**: as smartphones began to have reasonably good cameras, applications for scanning business cards and simple text started to appear, although their accuracy and speed were still limited compared to dedicated scanners.

## From 2012 Onward: The Deep Learning Revolution

**AlexNet's** victory in the 2012 ImageNet competition created a wave of deep learning that spread to every area of computer vision, and OCR was no exception. **Convolutional neural networks (CNNs)** replaced hand-crafted features for extracting character images. For long text sequences, models combining **RNN/LSTM with the CTC (Connectionist Temporal Classification) loss function** solved the alignment problem between images and character sequences without requiring manual per-character segmentation. A prime example is the **CRNN (Convolutional Recurrent Neural Network)** architecture published by Shi et al. in 2015, which became the foundation for countless scene text recognition systems afterward.

In parallel with recognition, the problem of **text detection in natural images** — that is, identifying which regions of an image contain text before passing them to recognition — also developed strongly, with models such as **EAST** (2017), **CRAFT** (Character Region Awareness for Text detection, 2019), and **DBNet** (Differentiable Binarization, 2019–2020), helping OCR handle difficult cases such as signs, tilted photographs, and curved text well.

## From 2017 Onward: Attention, Transformer, and Document AI

The **attention** mechanism and the **Transformer** architecture (introduced in 2017) were quickly incorporated into OCR, allowing models to "focus" on different parts of the image while decoding each character, significantly improving accuracy for long text and complex layouts.

The field of **Document AI** (understanding structured documents) also exploded:
- **LayoutLM** (Microsoft, 2020) is the pioneering model combining textual content, layout position, and image information to understand documents such as invoices and forms.
- **TrOCR** (Microsoft, 2021) applies an end-to-end Transformer architecture (image encoder – text decoder) to the OCR problem, simplifying the traditional pipeline, which required multiple separate steps (detection, recognition, post-processing).
- **Donut** (2022, developed by Naver) is an important turning point in the **"OCR-free document understanding"** direction — a model that understands document content directly from the image without a separate, explicit OCR step, foreshadowing the later trend.

## Important Commercial and Open-Source Tools and Platforms

Alongside academic research, the applied OCR market has also developed vigorously: **Google Cloud Vision OCR**, **AWS Textract**, and **Microsoft Azure Form Recognizer** (now **Azure AI Document Intelligence**) are popular cloud services for enterprises. On the open-source side, **PaddleOCR**, released by Baidu in 2020, stands out for its strong multilingual support (including Vietnamese), being lightweight and easy to deploy, making it a top choice for many developers.

## 2023–2026: The Era of General-Purpose Multimodal Language Models

The most recent period has seen the rise of **multimodal large language models (multimodal LLMs)** such as **GPT-4V**, **Gemini**, and **Qwen-VL**. These models are not specifically trained for OCR but have the ability to "read and understand" images containing text as a supplementary capability, even reading tables, charts, and handwriting in complex contexts — something traditional OCR usually has to handle through multiple separate steps.

At the same time, new dedicated OCR/document-understanding models continue to emerge to compete on accuracy and performance, such as **Kosmos-2.5** (Microsoft), **Nougat** (Meta, specialized for reading academic documents with math formulas), and **GOT-OCR2.0** — marketed as a "generation 2.0" OCR model unifying many tasks (reading printed text, formulas, tables, musical notation, etc.) within a single architecture. The overall trend of this period is **"OCR-free document understanding"** — gradually blurring the boundary between the "character recognition" step and the "content understanding" step, moving toward end-to-end systems that both read and reason over documents.

## Conclusion

From Fournier d'Albe's optophone converting light into sound more than a century ago, to today's massive multimodal models that can "read" and "understand" documents simultaneously, OCR has traveled a long road that mirrors the very history of artificial intelligence's development: from mechanical-optical devices, through pattern matching and statistics, to deep learning, and finally to general-purpose foundation models. Interestingly, the technology's original and most humane goal — helping visually impaired people "read" written text — has always remained a thread running throughout, from the 1976 Kurzweil Reading Machine to today's multimodal AI assistants that can describe and read text in images.
