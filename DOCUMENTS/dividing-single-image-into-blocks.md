```text
   DOCUMENTS/dividing-single-image-into-blocks.md
   Written by, Sohail Qayum Malik
```

> **Note to Readers:** This document is a work in progress, part of an ongoing series on a custom C++ transformer implementation. It extends the concepts introduced in Chapter 1, focusing on multi-head attention. Expect minor typos or formatting issues, which will be refined in future revisions. Thank you for your patience.

`"Readers should be aware that this article represents an ongoing project. The information and code contained herein are preliminary and will be expanded upon in future revisions."`

### I-JEPA Image Blocking Strategy Documentation

**Overview**

This document details the implementation of dividing single image into non-overlapping blocks for I-JEPA training, featuring consistent block sizing and random context-target assignment.

```Rust
/*
 * Calculate the step size for strided data access along rows
 * 
 * The step represents how many elements to skip between selections when
 * extracting data from the original buffer. It determines the compression
 * ratio of the resulting slice.
 * 
 * Formula: step = total_range_length / output_columns
 * Where:
 *   - total_range_length = (end - start) = number of elements in the source range
 *   - output_columns = shape.columns() = number of columns in the output slice
 *   - shape: It is the shape of each block/patch, its width (columns) and height (rows) 
 * Interpretation:
 * - step = 1: No compression, output uses all consecutive elements
 * - step > 1: Compression, output skips (step-1) elements between selections
 * - The step ensures the output has exactly 'shape.columns()' elements
 *   spread across the range from 'start' to 'end'
 */ 
let step = ((end as usize) - (start as usize))/(shape.columns() as usize);
```

```C++
/*
 * - self.shape: It is the shape of each whole mage which is then divided further into blocks/patches. This composite holds its width (columns) and height (rows)
 * - shape: It is the shape of each block/patch, its width (columns) and height (rows)    	
 * Before taking out patches/blocks from the image of (dimensions self.shape.as_ref().unwrap().get_columns() x self.shape.as_ref().unwrap().get_rows()) to columns x shape.get_rows() and then a single block at a time of dimensions (shape.get_columns() x shape.get_rows())
 */
```

```Rust
/*
 * Calculate the number of columns based on the compressed step size and buffer dimensions
 * 
 * This calculation determines how many data points will be displayed per row in the output,
 * accounting for the strided access pattern defined by the step size.
 * 
 * Breakdown of the calculation:
 * 1. self.data.as_ref().unwrap().len() - Total number of elements in the source buffer
 * 2. / step - Divide by step size to get the number of accessible elements after striding
 * 3. / (shape.rows() as usize) - Divide by number of rows to get columns per row
 * 
 * The result represents:
 * - The effective number of data columns after applying the compression step
 * - How many data points will be shown in each row of the final display
 * - The horizontal resolution of the compressed data visualization
 * 
 * This ensures the 2D output shape (rows × columns) properly represents the
 * compressed version of the original 1D data buffer.
 */            
let columns = (self.data.as_ref().unwrap().len() / step) / (shape.rows() as usize);
```

```C++
/*
 * Random patches approach, which are of equal size and which are non-overlapping
 * ------------------------------------------------------------------------------
 * Dimensions of each block, the last block's end is always less than `columns` which gets calculated using the above Rust statement.
 * The inflated IDAT chunks are concatenated and then they're treated as one continuous stream before dividing it into 8 equal data chunks. The atream is assumed to be have dimensions of `columns x shape.rows()`.
 */
```
```Rust
#[macro_export]
macro_rules! image_block_slice_start {

    ($block_number: expr, $image_block_width: expr, $image_channels: expr) => {
        
        ((($block_number - 1) as usize)*($image_block_width as usize)*($image_channels as usize)) as f64
    };    
}

#[macro_export]
macro_rules! image_block_slice_end {

    ($block_number: expr, $image_block_width: expr, $image_channels: expr) => {

        ((($block_number as usize)*($image_block_width as usize)*($image_channels as usize)) - 0) as f64
    };    
}

#[macro_export]
macro_rules! image_block_height {

    ($input_len: expr, $channels: expr) => {
                
        (($input_len/$channels) as f64/((JEPA_NUMBER_OF_CONTEXT_BLOCKS + JEPA_NUMBER_OF_TARGET_BLOCKS)) as f64 / JEPA_IMAGES_ASPECT_RATIO).sqrt() as f64
    };
}

#[macro_export]
macro_rules! image_block_width {

    ($input_len: expr, $channels: expr) => {
                
        (($input_len/$channels) as f64/((JEPA_NUMBER_OF_CONTEXT_BLOCKS + JEPA_NUMBER_OF_TARGET_BLOCKS)) as f64 / JEPA_IMAGES_ASPECT_RATIO).sqrt() as f64 * JEPA_IMAGES_ASPECT_RATIO
    };
}

#[macro_export]
macro_rules! image_block_size {

    ($input_len: expr, $channels: expr) => {
                        
        ($input_len/$channels)/(JEPA_NUMBER_OF_CONTEXT_BLOCKS + JEPA_NUMBER_OF_TARGET_BLOCKS)
    };
}

let image_block = ImageBlock::new (
    image_block_height! (input_pipeline_slice.data.as_ref().unwrap().len(), image_data_tensor_shape.get_channels()),
    image_block_width! (input_pipeline_slice.data.as_ref().unwrap().len(), image_data_tensor_shape.get_channels()),
    image_block_size! (input_pipeline_slice.data.as_ref().unwrap().len(), image_data_tensor_shape.get_channels())
);

let dims_image_block = Box::new(Dimensions::new(image_block.get_width(), image_block.get_height()));

let mut random_context_target_block_numbers: Box<[u8]> = Box::new([0; JEPA_NUMBER_OF_CONTEXT_BLOCKS + JEPA_NUMBER_OF_TARGET_BLOCKS]);

for j in 0..random_context_target_block_numbers.len() {
   let mut random_number: u8;
    
   loop {

       random_number = random_whole_number(1, random_context_target_block_numbers.len()) as u8;
       let mut is_duplicate = false;
        
       // Check all previous elements
       for k in 0..j {
           if random_number == random_context_target_block_numbers[k] {
               is_duplicate = true;
               break;
           }
      }
        
      if !is_duplicate {
          break;
      }
  }     
  random_context_target_block_numbers[j] = random_number;
}

for j in 0..random_context_target_block_numbers.len() {

   let image_block_slice: Box<Collective<T>> = input_pipeline_slice.get_slice(
       image_block_slice_start!(random_context_target_block_numbers[j], image_block.get_width(), image_data_tensor_shape.get_channels()),
       image_block_slice_end!(random_context_target_block_numbers[j], image_block.get_width(), image_data_tensor_shape.get_channels()),
       &dims_image_block,
       Axis::Rows
   );
```

#### Practical Example

8-Block Partitioning for a sample image processing scenario:

```C++
/*
 * The blocking strategy is consistent across entire dataset or all images: Please go through code above...
 */
```
```text
Block 1 always starts and end @ 0, 270, 120 strides starting from 0 inclusive to 270 exclusive = 32400 -> ( 120x(90x3) ) -> for 90x120 image block with three channels for each pixel
Block 2 always starts and end @ 270, 540, 120 strides starting from 270 inclusive to 540 exclusive = 32400  -> ( 120x(90x3) ) -> for 90x120 image block with three channels for each pixel
Block 3 always starts and end @ 540, 810, 120 strides starting from 540 inclusive to 810 exclusive = 32400 -> ( 120x(90x3) ) -> for 90x120 image block with three channels for each pixel
Block 4 always starts and end @ 810, 1080, 120 strides starting from 810 inclusive to 1080 exclusive = 32400 -> ( 120x(90x3) ) -> for 90x120 image block with three channels for each pixel 
Block 5 always starts and end @ 1080, 1350, 120 strides starting from 1080 inclusive to 1350 exclusive = 32400 -> ( 120x(90x3) ) -> for 90x120 image block with three channels for each pixel 
Block 6 always starts and end @ 1350, 1620, 120 strides starting from 1350 inclusive to 1620 exclusive = 32400 -> ( 120x(90x3) ) -> for 90x120 image block with three channels for each pixel 
Block 7 always starts and end @ 1620, 1890, 120 strides starting from 1620 inclusive to 1890 exclusive = 32400 -> ( 120x(90x3) ) -> for 90x120 image block with three channels for each pixel
Block 8 always starts and end @ 1890, 2160, 120 strides starting from 1890 inclusive to 2160 exclusive = 32400 -> ( 120x(90x3) ) -> for 90x120 image block with three channels for each pixel
```

Each block contains exactly 32,400 elements (90 width × 120 height × 3 channels).

**Validation check**:

If we can programmatically reconstruct the original image from all of its patches or blocks (even if it looks messy to us), then our blocking is correct for JEPA.
 - Allocate enough memory: 8 x ((shape.columns() x shape.rows()) x 3).
 - Treat this newly created mmemory to have dimensions dimensions (columns x shape.rows())
 - Now using above detail copy each block in the neswly allocated memory.


#### The blocking strategy must be consistent across all images in your dataset or a batch

**JEPA operates on embeddings, not pixels**:

In JEPA context, the blocks do NOT need to look like recognizable image parts to human eyes! The individual blocks can look like random noise to humans!
 - The model learns latent representations, not pixel-level reconstructions
 - Each block gets encoded into an embedding vector
 - The predictor learns relationships between these embeddings through context_embeddings model learns about target_embeddings

**What matters for JEPA**:
 
 - Spatial correspondence: Block 1 always comes from the same image region
 - Consistent partitioning: Same blocking strategy across all images
 - Information preservation: All original data is captured somewhere

#### Edge Byte Management Strategy

Only bytes at the **very bottom of the image** are excluded from blocking and this is due to enforcing uniform block sizes across all images.




