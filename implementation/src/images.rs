/*
 * implementation/src/images.rs
 * Q@khaa.pk
 */
 
use std::{rc::Rc, cell::RefCell};
use crate::{sundry::random_whole_number, constants::{NUMBER_OF_CONTEXT_BLOCKS, NUMBER_OF_TARGET_BLOCKS}};
use Numrs::{dimensions::Dimensions, collective::Collective, num::Tensor};

/// Configuration structure for machine learning model hyperparameters.
/// 
/// This struct encapsulates the essential training parameters needed to configure
/// a machine learning model. It provides a centralized way to manage and pass
/// around training configuration settings.
/// 
/// # Fields
/// * `learning_rate` - Controls how much to change the model in response to estimated error
/// * `batch_size` - Number of training examples processed before model is updated
/// * `epochs` - Number of complete passes through the training dataset
/// 
/// # Example
/// ```rust
/// let config = ModelConfig::new(0.001, 32, 100);
/// println!("Learning rate: {}", config.get_learning_rate());
/// ``` 
#[derive(Debug, Clone, Copy)]
pub struct ModelConfig {

    learning_rate: f64,
    batch_size: usize,
    epochs: usize,

    //pub hidden_layers: Vec<usize>,
}

/// Implementation block for ModelConfig providing constructor and accessor methods.
/// 
/// This implementation provides safe access to the private fields of ModelConfig
/// through getter methods, following Rust's encapsulation principles. The constructor
/// ensures proper initialization of all required parameters.
/// 
/// # Methods
/// * `new()` - Creates a new ModelConfig instance with specified parameters
/// * `get_learning_rate()` - Returns the configured learning rate
/// * `get_batch_size()` - Returns the configured batch size  
/// * `get_epochs()` - Returns the configured number of epochs
impl ModelConfig {
    
    pub fn new(learning_rate: f64, batch_size: usize, epochs: usize) -> ModelConfig {

        ModelConfig {

            learning_rate: learning_rate,
            batch_size: batch_size,
            epochs: epochs,
        }
    }

    pub fn get_learning_rate(&self) -> f64 {

        self.learning_rate
    }

    pub fn get_batch_size(&self) -> usize {

        self.batch_size
    }

    pub fn get_epochs(&self) -> usize {

        self.epochs
    }
}

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

/// Main model structure that combines configuration and data shape information.
/// 
/// This struct represents a machine learning model that integrates both the training
/// configuration parameters and the expected input data structure. It serves as the
/// primary interface for model operations and maintains the relationship between
/// model hyperparameters and data specifications.
/// 
/// # Fields
/// * `model_config` - The training configuration and hyperparameters
/// * `image_data_tensor_shape` - The expected shape of input image data
/// 
/// # Usage
/// This struct is designed to be the main entry point for model operations,
/// providing a unified interface that ensures consistency between model configuration
/// and expected data format.
/// 
/// # Example
/// ```rust
/// let config = ModelConfig::new(0.001, 32, 100);
/// let shape = ImageDataTensorShape::new(224, 224, 3);
/// let model = Model::new(config, shape);
/// model.start_training_loop();
/// ```
pub struct Model {
    
    model_config: ModelConfig,
    image_data_tensor_shape: ImageDataTensorShape,    
}

/// Implementation block for the Model struct providing core model functionality.
/// 
/// This implementation defines the behavior and methods available for the Model struct.
/// It includes constructor methods, accessors for the embedded configuration and shape
/// information, and placeholder methods for model operations like training.
/// 
/// # Methods
/// * `new()` - Constructs a new Model instance with given configuration and shape
/// * `get_ModelConfig()` - Returns a copy of the model's configuration
/// * `get_ImageDataTensorShape()` - Returns a copy of the expected input data shape
/// * `start_training_loop()` - Initiates the model training process (placeholder)
/// 
/// # Design Notes
/// The implementation uses cloning for the constructor parameters to ensure
/// the Model owns its configuration data, preventing issues with borrowed data
/// and enabling flexible usage patterns.
impl Model {

    pub fn new(model_config: ModelConfig, image_data_tensor_shape: ImageDataTensorShape) -> Model {

        Model {
            
           model_config: model_config.clone(),
           image_data_tensor_shape: image_data_tensor_shape.clone(),
        }
    }

    pub fn create_input_pipeline_old_old(&self, input_data_tensor_shape_format: ImageDataTensorShapeFormat) -> Box<Dimensions> {
    
        // Create a proper 3D tensor shape for image data: [batch, channels, height, width]
        // For JEPA, typically we'd have something like: batch_size -> channels -> height*width
    
        // Example: 32 batches -> 3 channels -> 224x224 pixels

        let batch_dim: Dimensions = Dimensions::new(0, self.model_config.get_batch_size()); // batch size
        let channel_dim: Dimensions;
        let height_dim: Dimensions; 

        if input_data_tensor_shape_format == ImageDataTensorShapeFormat::HWC {
            // If HWC format is specified, we need to convert it to CHW
            // This is a placeholder for conversion logic if needed 
            /*channel_dim = Dimensions::new(0, self.image_data_tensor_shape.get_channels())
                .with_next(Rc::new(RefCell::new(Dimensions::new(0, self.image_data_tensor_shape.get_height())
                    .with_next(Rc::new(RefCell::new(Dimensions::new(0, self.image_data_tensor_shape.get_width())))))));*/

            channel_dim = Dimensions::new( self.image_data_tensor_shape.get_channels(), self.image_data_tensor_shape.get_width() );
            height_dim = Dimensions::new(0, self.image_data_tensor_shape.get_height());
                            

            batch_dim.with_next(Rc::new(RefCell::new(height_dim)));
            //height_dim.with_prev(Rc::new(RefCell::new(batch_dim)));


            // Convert HWC to CHW
            // This is a placeholder for actual conversion logic if needed
            // In practice, you would need to rearrange the dimensions accordingly     

        } else if input_data_tensor_shape_format == ImageDataTensorShapeFormat::CHW {

            // If CHW format is specified, we can proceed directly
            
        } else {
            // Handle unsupported formats
            panic!("Unsupported ImageDataTensorShapeFormat");
        }


        /*let pixel_dims = Dimensions::new(224, 224);  // width=224, height=224
        let channel_dims = Dimensions::new(0, 3)     // 3 channels (RGB)
            .with_next(Rc::new(RefCell::new(pixel_dims)));
        let batch_dims = Dimensions::new(0, 32)      // batch size of 32
            .with_next(Rc::new(RefCell::new(channel_dims)));
    
        //Box::new(batch_dims)*/

        Box::new(Dimensions::new(0, 0)) // Placeholder for actual tensor shape creation
        
    }

    // Version with both next AND prev pointers (doubly-linked)
    pub fn create_input_pipeline_with_prev(&self, input_data_tensor_shape_format: ImageDataTensorShapeFormat) -> Box<Dimensions> {
    
        let batch_size = self.model_config.get_batch_size();
        let channels = self.image_data_tensor_shape.get_channels();
        let height = self.image_data_tensor_shape.get_height();
        let width = self.image_data_tensor_shape.get_width();

        match input_data_tensor_shape_format {
            ImageDataTensorShapeFormat::CHW => {
                // Create the dimensions first
                let width_dim = Dimensions::new(width, height);
                let channel_dim = Dimensions::new(0, channels);
                let batch_dim = Dimensions::new(0, batch_size);
            
                // Wrap them in Rc<RefCell<>> for sharing
                let width_rc = Rc::new(RefCell::new(width_dim));
                let channel_rc = Rc::new(RefCell::new(channel_dim));
                let batch_rc = Rc::new(RefCell::new(batch_dim));
            
                // Set up the forward links (next)
                batch_rc.borrow_mut().set_next(Some(channel_rc.clone()));
                channel_rc.borrow_mut().set_next(Some(width_rc.clone()));
            
                // Set up the backward links (prev)
                channel_rc.borrow_mut().set_prev(Some(batch_rc.clone()));
                width_rc.borrow_mut().set_prev(Some(channel_rc.clone()));
            
                // Extract the root dimension (avoid borrow checker issues)
                let result = {
                    let borrowed = batch_rc.borrow();
                    borrowed.clone()
                };
            
                Box::new(result)
            },
        
            ImageDataTensorShapeFormat::HWC => {
                // Create the dimensions
                let channel_dim = Dimensions::new(channels, width);
                let height_dim = Dimensions::new(0, height);
                let batch_dim = Dimensions::new(0, batch_size);
            
                // Wrap them in Rc<RefCell<>>
                let channel_rc = Rc::new(RefCell::new(channel_dim));
                let height_rc = Rc::new(RefCell::new(height_dim));
                let batch_rc = Rc::new(RefCell::new(batch_dim));
            
                // Set up forward links
                batch_rc.borrow_mut().set_next(Some(height_rc.clone()));
                height_rc.borrow_mut().set_next(Some(channel_rc.clone()));
            
                // Set up backward links
                height_rc.borrow_mut().set_prev(Some(batch_rc.clone()));
                channel_rc.borrow_mut().set_prev(Some(height_rc.clone()));
            
                // Extract result
                let result = {
                    let borrowed = batch_rc.borrow();
                    borrowed.clone()
                };
            
                Box::new(result)
            },
        
            _ => {
                // Default case - simple 2D
                Box::new(Dimensions::new(width, height))
            }
        }
    }
    
    // Helper function to add prev pointers to an existing chain
    fn add_prev_pointers(mut root: Dimensions) -> Dimensions {
        let mut current_opt = Some(Rc::new(RefCell::new(root.clone())));
        let mut prev_rc: Option<Rc<RefCell<Dimensions>>> = None;
    
        while let Some(current_rc) = current_opt {
            // Set prev pointer if we have a previous node
            if let Some(prev) = &prev_rc {
                current_rc.borrow_mut().set_prev(Some(prev.clone()));
            }
        
            // Move to next node
            let next_opt = current_rc.borrow().next();
            prev_rc = Some(current_rc);
            current_opt = next_opt;
        }
    
        root
    }

    // BETTER APPROACH: Use the builder pattern from your Dimensions struct
    pub fn create_input_pipeline_builder_pattern(&self, input_data_tensor_shape_format: ImageDataTensorShapeFormat) -> Box<Dimensions> {
    
        let batch_size = self.model_config.get_batch_size();
        let channels = self.image_data_tensor_shape.get_channels();
        let height = self.image_data_tensor_shape.get_height();
        let width = self.image_data_tensor_shape.get_width();

        match input_data_tensor_shape_format {
            ImageDataTensorShapeFormat::CHW => {
                // Build using the fluent interface - much cleaner!
                let width_dim = Dimensions::new(width, height);
                let channel_dim = Dimensions::new(0, channels)
                    .with_next(Rc::new(RefCell::new(width_dim)));
                let batch_dim = Dimensions::new(0, batch_size)
                    .with_next(Rc::new(RefCell::new(channel_dim)));

                // Now add prev pointers
                let batch_dim = Self::add_prev_pointers(batch_dim);
                                               
                Box::new(batch_dim)
            },
        
            ImageDataTensorShapeFormat::HWC => {
                let channel_dim = Dimensions::new(channels, width);
                let height_dim = Dimensions::new(0, height)
                    .with_next(Rc::new(RefCell::new(channel_dim)));
                let batch_dim = Dimensions::new(0, batch_size)
                    .with_next(Rc::new(RefCell::new(height_dim)));

                // Add prev pointers
                let batch_dim = Self::add_prev_pointers(batch_dim);
            
                Box::new(batch_dim)
            },
        
            _ => {
                Box::new(Dimensions::new(width, height))
            }
        }
    }
    
    // Simple version without prev pointers (your current approach)
    pub fn create_input_pipeline_simple(&self, input_data_tensor_shape_format: ImageDataTensorShapeFormat) -> Box<Dimensions> {
    
        let batch_size = self.model_config.get_batch_size();
        let channels = self.image_data_tensor_shape.get_channels();
        let height = self.image_data_tensor_shape.get_height();
        let width = self.image_data_tensor_shape.get_width();

        match input_data_tensor_shape_format {
            ImageDataTensorShapeFormat::CHW => {
                let width_dim = Dimensions::new(width, height);
                let channel_dim = Dimensions::new(0, channels)
                    .with_next(Rc::new(RefCell::new(width_dim)));
                let batch_dim = Dimensions::new(0, batch_size)
                    .with_next(Rc::new(RefCell::new(channel_dim)));
            
                Box::new(batch_dim)
            },
        
            ImageDataTensorShapeFormat::HWC => {
                let channel_dim = Dimensions::new(channels, width);
                let height_dim = Dimensions::new(0, height)
                    .with_next(Rc::new(RefCell::new(channel_dim)));
                let batch_dim = Dimensions::new(0, batch_size)
                    .with_next(Rc::new(RefCell::new(height_dim)));
            
                Box::new(batch_dim)
            },
        
            _ => {
                Box::new(Dimensions::new(width, height))
            }
        }
    }

    pub fn get_ModelConfig(&self) -> ModelConfig {

        self.model_config
    }

    pub fn get_ImageDataTensorShape(&self) -> ImageDataTensorShape {

        self.image_data_tensor_shape
    }

    /// Initiates the JEPA (Joint Embedding Predictive Architecture) training process.
    /// 
    /// This method serves as the main entry point for training the self-supervised learning
    /// model. It orchestrates the entire training pipeline including data preprocessing,
    /// forward passes through context and target encoders, predictor network computation,
    /// joint embedding loss calculation, and parameter updates.
    /// 
    /// # Parameters
    /// * `image_data_tensor_shape_format` - Specifies the memory layout format of input tensors
    ///   - `CHW`: Optimized for GPU operations and convolutional processing
    ///   - `HWC`: Compatible with standard image libraries and CPU processing
    /// 
    /// # JEPA Training Pipeline Overview
    /// 
    /// ## Phase 1: Data Preprocessing
    /// - Loads and preprocesses image batches according to specified tensor format
    /// - Applies data augmentations and normalization
    /// - Converts between tensor formats if necessary (HWC ↔ CHW)
    /// 
    /// ## Phase 2: Masking Strategy
    /// - Generates random context and target block selections
    /// - Creates masking patterns for self-supervised learning
    /// - Ensures non-overlapping context and target regions
    /// 
    /// ## Phase 3: Encoder Forward Passes
    /// - **Context Encoder**: Processes visible image patches to generate context representations
    /// - **Target Encoder**: Processes target patches with stop-gradient (EMA updates)
    /// - Maintains representation consistency across different views
    /// 
    /// ## Phase 4: Prediction and Loss Computation
    /// - **Predictor Network**: Maps context representations to target representation space
    /// - **Joint Embedding Loss**: Measures similarity between predicted and actual target embeddings
    /// - Avoids pixel-level reconstruction, focusing on semantic representations
    /// 
    /// ## Phase 5: Optimization
    /// - Backpropagates gradients through context encoder and predictor
    /// - Updates target encoder via exponential moving average (EMA)
    /// - Applies configured optimizer (Adam, SGD, etc.) with specified learning rate
    /// 
    /// # Format-Specific Optimizations
    /// 
    /// ## CHW Format Processing
    /// - Utilizes vectorized operations for channel-wise processing
    /// - Optimizes memory access patterns for convolutional operations
    /// - Enables efficient GPU kernel execution
    /// 
    /// ## HWC Format Processing  
    /// - Handles pixel-interleaved data efficiently
    /// - Performs format conversion when necessary for ML operations
    /// - Maintains compatibility with standard image processing workflows
    /// 
    /// # Training Loop Structure
    /// ```text
    /// for epoch in 0..config.epochs {
    ///     for batch in data_loader {
    ///         1. Preprocess batch according to tensor format
    ///         2. Generate masking strategy
    ///         3. Forward pass: context_encoder(masked_input)
    ///         4. Forward pass: target_encoder(target_patches) 
    ///         5. Forward pass: predictor(context_embeddings)
    ///         6. Compute joint embedding loss
    ///         7. Backward pass and parameter updates
    ///         8. Update target encoder via EMA
    ///     }
    /// }
    /// ```
    /// 
    /// # Performance Considerations
    /// - **Memory Efficiency**: Tensor format affects memory access patterns and cache utilization
    /// - **GPU Utilization**: CHW format typically provides better GPU throughput
    /// - **Batch Processing**: Leverages configured batch size for optimal hardware utilization
    /// 
    /// # Implementation Status
    /// **Note**: This method currently serves as a placeholder for the complete training
    /// implementation. The full JEPA training logic including encoder architectures,
    /// masking strategies, and loss computations will be implemented in subsequent phases.
    /// 
    /// # Example Usage
    /// ```rust
    /// let config = ModelConfig::new(0.0001, 64, 100);
    /// let shape = ImageDataTensorShape::new(3, 224, 224);
    /// let model = Model::new(config, shape);
    /// 
    /// // Start training with CHW format (recommended for performance)
    /// model.start_training_loop(ImageDataTensorShapeFormat::CHW);
    /// ```    
    pub fn start_training_loop<T>(&self, input_pipeline: &Collective<T>, image_data_tensor_shape_format: ImageDataTensorShapeFormat) 
        where 
            T: Default + Copy 
    {        
        let model_config = self.get_ModelConfig();
        let image_data_tensor_shape = self.get_ImageDataTensorShape();

        //let random_context_target_block_numbers: u8[NUMBER_OF_CONTEXT_BLOCKS + NUMBER_OF_TARGET_BLOCKS];

        let mut random_context_target_block_numbers: Box<[u8]> = Box::new([0; NUMBER_OF_CONTEXT_BLOCKS + NUMBER_OF_TARGET_BLOCKS]);

        model_config.get_epochs();
        model_config.get_batch_size();
        model_config.get_learning_rate();
        image_data_tensor_shape.get_channels();
        image_data_tensor_shape.get_height();
        image_data_tensor_shape.get_width();

        //println! ("{}", image_data_tensor_shape.get_channels()*image_data_tensor_shape.get_height()*image_data_tensor_shape.get_width());
        //println! ("{}", model_config.get_batch_size());

        let dims = Box::new(Dimensions::new(image_data_tensor_shape.get_width(), image_data_tensor_shape.get_height()));
        
        //let input_pipeline_slice: Box<Collective<T>> = input_pipeline.get_slice(image_data_tensor_shape.get_channels()*image_data_tensor_shape.get_height()*image_data_tensor_shape.get_width(), image_data_tensor_shape.get_channels()*image_data_tensor_shape.get_height()*image_data_tensor_shape.get_width()*2, dims);

        //println! ("{}", input_pipeline_slice.data.unwrap().len());

        for i in 0..model_config.get_batch_size() {

            let input_pipeline_slice: Box<Collective<T>> = input_pipeline.get_slice(image_data_tensor_shape.get_channels()*image_data_tensor_shape.get_height()*image_data_tensor_shape.get_width()*i, image_data_tensor_shape.get_channels()*image_data_tensor_shape.get_height()*image_data_tensor_shape.get_width()*(i+1), dims.clone());

            let random_number: u8 = random_whole_number(1, random_context_target_block_numbers.len()) as u8;

            println! ("random number = {}", random_number); 

            println!("len = {}", random_context_target_block_numbers.len());

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
             
                println! ("-> {}", random_context_target_block_numbers[j]);
            }
        }
    }   
}
