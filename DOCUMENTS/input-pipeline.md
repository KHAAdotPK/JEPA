```C++
    JEPA/DOCUMENTS/input-pipeline.md
    Written by, Sohail Qayum Malik
```
> **Note to Readers:** This document is a work in progress, part of an ongoing series on a custom C++ transformer implementation. It extends the concepts introduced in Chapter 1, focusing on multi-head attention. Expect minor typos or formatting issues, which will be refined in future revisions. Thank you for your patience.

`"Readers should be aware that this article represents an ongoing project. The information and code contained herein are preliminary and will be expanded upon in future revisions."`

# I-JEPA Input Pipeline Documentation  

## Overview
This document describes the custom input pipeline for an Image-based Joint-Embedding Predictive Architecture (I-JEPA) implementation in Rust. The pipeline features a unique approach to PNG processing using a custom library rather than standard image decoding utilities.

## Architecture Components

### Custom PNG Processing
**Library**: [PNG-rust](https://github.com/KHAAdotPK/PNG-rust)
- **Approach**: Direct IDAT chunk manipulation rather than conventional PNG decoding
- **Implementation**: Custom Rust library with C bindings
- **Process Flow**:
  1. Extract all IDAT chunks from original PNG files
  2. Concatenate IDAT chunks per image
  3. Inflate compressed data to raw pixel format

### Block Generation Strategy
- **Total Blocks**: (`JEPA_NUMBER_OF_CONTEXT_BLOCKS` **+** `JEPA_NUMBER_OF_TARGET_BLOCKS`) blocks per image total
- **Distribution**: Half context blocks + Half target blocks (50/50 split)
- **Block Properties**:
  - Consistent aspect ratio across all blocks (`JEPA_IMAGES_ASPECT_RATIO`)
  - Identical pixel dimensions for all blocks
  - Non-overlapping between context and target sets
  - Random spatial sampling within image bounds

### Data Flow
```
PNG Files → IDAT Extraction → Concatenation → Inflation → 
Image Division → Block Sampling → Context/Target Separation
```

## Key Design Decisions

### Block Sampling
- **Random Selection**: Context and target blocks randomly sampled each epoch
- **Spatial Constraints**: Blocks maintain natural aspect ratios
- **Size Calculation**: Based on dividing total image area by (`JEPA_NUMBER_OF_CONTEXT_BLOCKS` **+** `JEPA_NUMBER_OF_TARGET_BLOCKS`) blocks
- **Non-overlap Guarantee**: Critical for preventing predictor cheating

### Custom PNG Handling
- **Rationale**: Full control over image decoding process
- **Advantage**: Potential performance optimization opportunities
- **Risk**: Assumes perfect equivalence with standard PNG decoders

## I-JEPA Alignment

### Core Principles Implemented
✅ **Context-Target Separation**: Clear distinction between visible and predicted regions  
✅ **Spatial Prediction**: Predictor must reason about relative positions  
✅ **Semantic Learning**: Forces abstraction beyond pixel-level features  
✅ **Data Efficiency**: Multiple learning signals per image (`JEPA_NUMBER_OF_CONTEXT_BLOCKS` **+** `JEPA_NUMBER_OF_TARGET_BLOCKS`)/2 predictions

### Training Readiness
The pipeline produces:
- Context blocks for encoder input
- Target blocks for prediction tasks
- Proper spatial relationships for positional encoding
- Augmented training signals through random sampling

#### Configuration Constants

| Constant | Description | Example Value |
|--------|-------------|---------------|
| `JEPA_NUMBER_OF_CONTEXT_BLOCKS` | Number of visible context blocks | 4 |
| `JEPA_NUMBER_OF_TARGET_BLOCKS` | Number of blocks to predict | 4 |
| `JEPA_IMAGES_ASPECT_RATIO` | Enforced aspect ratio (width/height) | 0.75 |


#### Block Size Calculation
Blocks are sampled **without replacement** and **without spatial overlap**.

- Let:
  - `W, H` = image width, height (in pixels)
  - `N = JEPA_NUMBER_OF_CONTEXT_BLOCKS + JEPA_NUMBER_OF_TARGET_BLOCKS`
  - `A = W × H` (total pixel area)
  - `r = JEPA_IMAGES_ASPECT_RATIO`

- **Target area per block**:  
  $$ a = \frac{A}{N} $$

- **Block dimensions (floating-point)**:  
  $$ h = \sqrt{\frac{a}{r}},\quad w = r \times h $$

  → Both `w` and `h` are **`f64`** and **preserve fractional precision**

## Next Steps
This pipeline provides a solid foundation for implementing the core I-JEPA components:
- Vision Transformer encoder
- Spatial-aware predictor
- Momentum encoder with stop-gradient
- Representation similarity loss

The custom PNG processing approach offers a unique implementation path while maintaining the essential I-JEPA learning dynamics.