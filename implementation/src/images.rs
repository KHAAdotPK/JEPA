/*
 * implementation/src/images.rs
 * Q@khaa.pk
 */
 
use std::{rc::Rc, cell::RefCell};
//use crate::{sundry::random_whole_number, constants::{NUMBER_OF_CONTEXT_BLOCKS, NUMBER_OF_TARGET_BLOCKS}};
//use Numrs::{dimensions::Dimensions, collective::Collective, num::Tensor};


// Flexibility to handle both formats if needed
/// Enumeration defining supported tensor data layout formats for image processing.
/// 
/// This enum provides flexibility in handling different tensor memory layouts that may be
/// required when interfacing with various image processing libraries, GPU frameworks, or
/// legacy systems. The choice of format significantly impacts memory access patterns,
/// performance characteristics, and compatibility with different computational backends.
/// 
/// # Variants
/// 
/// ## CHW (Channels, Height, Width) - **Recommended Default**
/// - **Memory Layout**: All pixels of channel 0, then all pixels of channel 1, etc.
/// - **Use Cases**: 
///   - Convolutional neural networks and Vision Transformers
///   - GPU-accelerated operations (CUDA, OpenCL)
///   - Deep learning frameworks (PyTorch, TensorFlow)
///   - JEPA patch extraction and processing
/// - **Performance**: Optimized for channel-wise operations and SIMD instructions
/// - **Batch Dimension**: Extends to NCHW (Batch, Channels, Height, Width)
/// 
/// ## HWC (Height, Width, Channels) - **Interoperability Format**
/// - **Memory Layout**: Interleaved channels (R,G,B,R,G,B,...)
/// - **Use Cases**:
///   - OpenCV image processing operations
///   - Some image I/O libraries and codecs
///   - CPU-based image manipulation
///   - Interfacing with graphics APIs expecting interleaved data
/// - **Performance**: Better for pixel-wise operations but slower for ML computations
/// 
/// # Performance Implications
/// - **CHW**: Superior cache locality for convolutions and matrix operations
/// - **HWC**: Better for pixel-level processing but requires transposition for ML ops
/// 
/// # Example Usage
/// ```rust
/// // Using CHW for ML operations (recommended)
/// model.start_training_loop(ImageDataTensorShapeFormat::CHW);
/// 
/// // Using HWC when interfacing with OpenCV or similar libraries
/// model.start_training_loop(ImageDataTensorShapeFormat::HWC);
/// ```
/// 
/// # Implementation Notes
/// The format parameter allows the training loop to adapt its data preprocessing
/// and tensor operations based on the input format, ensuring optimal performance
/// regardless of the source data layout.
#[derive(PartialEq)]
pub enum ImageDataTensorShapeFormat {
    CHW,  // The primary choice
    HWC,  // For interfacing with certain image libraries
} 

/*
    Tensor Shape Decision for JEPA
    ------------------------------
    CHW (Channel, Height, Width) format is chosen for the following reasons:
    1. **Consistency with Convolutional Neural Networks (CNNs)**: Most CNN architectures expect input in CHW format, making it easier to integrate with existing models and libraries.
    2. **Memory Efficiency**: CHW format can be more memory-efficient for certain operations, especially when dealing with large images and multiple channels.
    3. **Performance Optimization**: Many deep learning frameworks optimize operations for CHW format, leading to better performance during training and inference.
    4. **Flexibility**: CHW format allows for easier manipulation of image data, such as cropping or resizing, while maintaining the integrity of the channel information.
    5. **GPU Compatibility**: Most deep learning frameworks expect CHW for convolutional operations.
    6. **Standardization**: CHW is a widely accepted standard in the machine learning community, ensuring compatibility with various datasets and pre-trained models.
    7. **Batch Processing**: When you scale up, batch dimension becomes (batch, channels, height, width).
 */

/// Represents the dimensional structure of image data tensors.
/// 
/// This struct defines the shape characteristics of image data used in machine learning
/// models, particularly for computer vision tasks. It encapsulates the three fundamental
/// dimensions that define image data structure.
/// 
/// # Fields
/// * `channels` - The number of color channels (e.g., 3 for RGB, 1 for grayscale)
/// * `height` - The vertical dimension of the image in pixels
/// * `width` - The horizontal dimension of the image in pixels  
/// 
/// # Example
/// ```rust
/// // RGB image of 224x224 pixels
/// let shape = ImageDataTensorShape::new(3, 224, 224);
/// 
/// // Grayscale image of 28x28 pixels (like MNIST)
/// let mnist_shape = ImageDataTensorShape::new(1, 28, 28);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ImageDataTensorShape {

    channels: usize,
    height: usize,
    width: usize,    
}

/// Implementation block for ImageDataTensorShape providing constructor and accessor methods.
/// 
/// This implementation ensures controlled access to the image tensor dimensions through
/// getter methods. It provides a clean interface for retrieving shape information that
/// can be used for tensor operations and model architecture configuration.
/// 
/// # Methods
/// * `new()` - Creates a new ImageDataTensorShape with specified dimensions
/// * `get_height()` - Returns the height dimension of the image tensor
/// * `get_width()` - Returns the width dimension of the image tensor
/// * `get_channels()` - Returns the number of channels in the image tensor
impl ImageDataTensorShape {

    pub fn new(channels: usize, height: usize, width: usize) -> ImageDataTensorShape {

        ImageDataTensorShape {

            channels: channels,
            height: height,
            width: width,            
        }
    }

    pub fn get_height(&self) -> usize {

        self.height
    }

    pub fn get_width(&self) -> usize {

        self.width
    }

    pub fn get_channels(&self) -> usize {

        self.channels
    }
}

