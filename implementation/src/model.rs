/*
 * implementation/src/model.rs
 * Q@khaa.pk
 */

use crate::{
    constants::{
        JEPA_IMAGES_ASPECT_RATIO, JEPA_IMAGE_BLOCK_FILE_NAME_EXTENSION,
        JEPA_IMAGE_BLOCK_FILE_NAME_POSTLUDE, JEPA_IMAGE_BLOCK_FILE_NAME_PRELUDE,
        JEPA_NUMBER_OF_CONTEXT_BLOCKS, JEPA_NUMBER_OF_TARGET_BLOCKS,
    }, /*, images::{ImageDataTensorShape, ImageDataTensorShapeFormat, ImageBlock}*/
    /*image_block_height,*/ /*image_block_height_vertical*//*, image_block_width*//*, image_block_width_vertical*//*, image_block_size*//*, image_block_size_vertical*//*, image_block_slice_start*//*, image_block_slice_start_vertical*//*, image_block_slice_end*//*, image_block_slice_end_vertical*//*, input_pipeline_slice_start*//*, input_pipeline_slice_end,*/
    sundry::random_whole_number,
};
use png::{
    constants::{PNG_FILE_EXTENSION, PNG_OUTPUT_FILE_SUFFIX},
    image_block_height, image_block_size, image_block_slice_end,
    image_block_slice_end_horizontal_experimental, image_block_slice_end_vertical_experimental,
    image_block_slice_start, image_block_slice_start_experimental,
    image_block_slice_start_horizontal_experimental, image_block_slice_start_vertical,
    image_block_slice_start_vertical_experimental, image_block_width,
    images::{ImageBlock, ImageDataTensorShape, ImageDataTensorShapeFormat},
    input_pipeline_slice_end, input_pipeline_slice_start,
    png_core::{create_png_from_collective, create_png_from_png_files, Png},
    zero_based_block_number_horizontal_experimental,
};
use std::{cell::RefCell, path::Path, rc::Rc};
use Numrs::{collective::Collective, dimensions::Dimensions, header::Axis, num::Tensor};

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

    pub fn create_input_pipeline_old_old(
        &self,
        input_data_tensor_shape_format: ImageDataTensorShapeFormat,
    ) -> Box<Dimensions> {
        // Create a proper 3D tensor shape for image data: [batch, channels, height, width]
        // For JEPA, typically we'd have something like: batch_size -> channels -> height*width

        // Example: 32 batches -> 3 channels -> 224x224 pixels

        let batch_dim: Dimensions = Dimensions::new(0.0, self.model_config.get_batch_size() as f64); // batch size
        let channel_dim: Dimensions;
        let height_dim: Dimensions;

        if input_data_tensor_shape_format == ImageDataTensorShapeFormat::HWC {
            // If HWC format is specified, we need to convert it to CHW
            // This is a placeholder for conversion logic if needed
            /*channel_dim = Dimensions::new(0, self.image_data_tensor_shape.get_channels())
            .with_next(Rc::new(RefCell::new(Dimensions::new(0, self.image_data_tensor_shape.get_height())
                .with_next(Rc::new(RefCell::new(Dimensions::new(0, self.image_data_tensor_shape.get_width())))))));*/

            channel_dim = Dimensions::new(
                self.image_data_tensor_shape.get_channels() as f64,
                self.image_data_tensor_shape.get_width(),
            );
            height_dim = Dimensions::new(0.0, self.image_data_tensor_shape.get_height());

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

        Box::new(Dimensions::new(0.0, 0.0)) // Placeholder for actual tensor shape creation
    }

    // Version with both next AND prev pointers (doubly-linked)
    pub fn create_input_pipeline_with_prev(
        &self,
        input_data_tensor_shape_format: ImageDataTensorShapeFormat,
    ) -> Box<Dimensions> {
        let mut batch_size = self.model_config.get_batch_size();
        let channels = self.image_data_tensor_shape.get_channels();
        let height = self.image_data_tensor_shape.get_height();
        let width = self.image_data_tensor_shape.get_width();

        match input_data_tensor_shape_format {
            ImageDataTensorShapeFormat::CHW => {
                // Create the dimensions first
                let width_dim = Dimensions::new(width, height);
                let channel_dim = Dimensions::new(0.0, channels as f64);
                let batch_dim = Dimensions::new(0.0, batch_size as f64);

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
            }

            ImageDataTensorShapeFormat::HWC => {
                // Create the dimensions
                let channel_dim = Dimensions::new(channels as f64, width);
                let height_dim = Dimensions::new(0.0, height);
                let batch_dim = Dimensions::new(0.0, batch_size as f64);

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
            }

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
    pub fn create_input_pipeline_builder_pattern(
        &self,
        input_data_tensor_shape_format: ImageDataTensorShapeFormat,
    ) -> Box<Dimensions> {
        let batch_size = self.model_config.get_batch_size();
        let channels = self.image_data_tensor_shape.get_channels() as f64;
        let height = self.image_data_tensor_shape.get_height();
        let width = self.image_data_tensor_shape.get_width();

        match input_data_tensor_shape_format {
            ImageDataTensorShapeFormat::CHW => {
                // Build using the fluent interface - much cleaner!
                let width_dim = Dimensions::new(width, height);
                let channel_dim =
                    Dimensions::new(0.0, channels).with_next(Rc::new(RefCell::new(width_dim)));
                let batch_dim = Dimensions::new(0.0, batch_size as f64)
                    .with_next(Rc::new(RefCell::new(channel_dim)));

                // Now add prev pointers
                let batch_dim = Self::add_prev_pointers(batch_dim);

                Box::new(batch_dim)
            }

            ImageDataTensorShapeFormat::HWC => {
                let channel_dim = Dimensions::new(channels as f64, width);
                let height_dim =
                    Dimensions::new(0.0, height).with_next(Rc::new(RefCell::new(channel_dim)));
                let batch_dim = Dimensions::new(0.0, batch_size as f64)
                    .with_next(Rc::new(RefCell::new(height_dim)));

                // Add prev pointers
                let batch_dim = Self::add_prev_pointers(batch_dim);

                Box::new(batch_dim)
            }

            _ => Box::new(Dimensions::new(width, height)),
        }
    }

    // Simple version without prev pointers (your current approach)
    pub fn create_input_pipeline_simple(
        &self,
        input_data_tensor_shape_format: ImageDataTensorShapeFormat,
    ) -> Box<Dimensions> {
        let batch_size = self.model_config.get_batch_size() as f64;
        let channels = self.image_data_tensor_shape.get_channels() as f64;
        let height = self.image_data_tensor_shape.get_height();
        let width = self.image_data_tensor_shape.get_width();

        match input_data_tensor_shape_format {
            ImageDataTensorShapeFormat::CHW => {
                let width_dim = Dimensions::new(width, height);
                let channel_dim =
                    Dimensions::new(0.0, channels).with_next(Rc::new(RefCell::new(width_dim)));
                let batch_dim =
                    Dimensions::new(0.0, batch_size).with_next(Rc::new(RefCell::new(channel_dim)));

                Box::new(batch_dim)
            }

            ImageDataTensorShapeFormat::HWC => {
                let channel_dim = Dimensions::new(channels, width);
                let height_dim =
                    Dimensions::new(0.0, height).with_next(Rc::new(RefCell::new(channel_dim)));
                let batch_dim =
                    Dimensions::new(0.0, batch_size).with_next(Rc::new(RefCell::new(height_dim)));

                Box::new(batch_dim)
            }

            _ => Box::new(Dimensions::new(width, height)),
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
    pub fn start_training_loop<T>(
        &self,
        input_pipeline: &Collective<T>,
        image_data_tensor_shape_format: ImageDataTensorShapeFormat,
        verbose: bool,
    ) where
        T: Default + Copy,
    {
        let model_config = self.get_ModelConfig();
        let image_data_tensor_shape = self.get_ImageDataTensorShape();

        // Automatically dropped when function returns
        let mut random_context_target_block_numbers: Box<[u8]> =
            Box::new([0; JEPA_NUMBER_OF_CONTEXT_BLOCKS + JEPA_NUMBER_OF_TARGET_BLOCKS]);

        model_config.get_epochs();
        model_config.get_batch_size();
        model_config.get_learning_rate();
        image_data_tensor_shape.get_channels();
        image_data_tensor_shape.get_height();
        image_data_tensor_shape.get_width();

        let mut block_file_names: Vec<String> = Vec::<String>::new();

        //println! ("{}", image_data_tensor_shape.get_channels()*(image_data_tensor_shape.get_height() as usize)*(image_data_tensor_shape.get_width() as usize));
        //println! ("{}", model_config.get_batch_size());

        let mut dims_image = Box::new(Dimensions::new(
            image_data_tensor_shape.get_width(),
            image_data_tensor_shape.get_height(),
        ));

        //println! ("{}, {}", dims_image.as_ref().get_width(), dims_image.as_ref().get_height());

        //let input_pipeline_slice: Box<Collective<T>> = input_pipeline.get_slice(image_data_tensor_shape.get_channels()*image_data_tensor_shape.get_height()*image_data_tensor_shape.get_width(), image_data_tensor_shape.get_channels()*image_data_tensor_shape.get_height()*image_data_tensor_shape.get_width()*2, dims);

        //println! ("{}", input_pipeline_slice.data.unwrap().len());

        for i in 0..model_config.get_batch_size() {
            // Automatically dropped at end of each loop iteration
            let input_pipeline_slice: Box<Collective<T>> = input_pipeline.get_slice (
                /*(image_data_tensor_shape.get_channels() as f64)*image_data_tensor_shape.get_height()*image_data_tensor_shape.get_width()*(i as f64),*/
                input_pipeline_slice_start! (image_data_tensor_shape.get_height(), image_data_tensor_shape.get_width(), image_data_tensor_shape.get_channels(), i),
                /*(image_data_tensor_shape.get_channels() as f64)*image_data_tensor_shape.get_height()*image_data_tensor_shape.get_width()*((i + 1) as f64),*/
                input_pipeline_slice_end! (image_data_tensor_shape.get_height(), image_data_tensor_shape.get_width(), image_data_tensor_shape.get_channels(), i),
                &dims_image,
                Axis::None
            );

            let mut path_text = format!("{}{}", i + 1, PNG_FILE_EXTENSION);
            let mut path = Path::new(&path_text);

            let mut png_png = create_png_from_collective::<T>(&input_pipeline_slice, &path);

            //png_png_png.save_to_file(&path_path);

            match png_png {
                Some(png) => {
                    let result = png.save_to_file(&path);

                    match result {
                        Ok(_) => {
                            if verbose {
                                println!("Saved PNG to file: \"{}\"", path.display());
                                println!("Traversing PNG of \"{}\" file:", path.display());
                                png.traverse();
                            }
                        }
                        Err(e) => {
                            println!("Model::start_training_loop() Error: {}", e);
                        }
                    }
                }
                None => {
                    println!("Failed to create PNG from Collective<T>");
                }
            }

            //let input_pipeline_slice: Box<Collective<T>> = input_pipeline.get_slice(image_data_tensor_shape.get_channels()*image_data_tensor_shape.get_height()*image_data_tensor_shape.get_width()*i, image_data_tensor_shape.get_channels()*image_data_tensor_shape.get_height()*image_data_tensor_shape.get_width()*(i+1), dims_image.clone(), Axis::None);

            /*let random_number: u8 = random_whole_number(1, random_context_target_block_numbers.len()) as u8;

            println! ("random number = {}", random_number);

            println!("len = {}", random_context_target_block_numbers.len());*/

            //input_pipeline_slice.get_slice(0, 10, dims.clone());

            //let image_block = ImageBlock::new(image_data_tensor_shape.get_height() as f64, image_data_tensor_shape.get_width() as f64, (input_pipeline_slice.data.as_ref().unwrap().len()/image_data_tensor_shape.get_channels())/8);

            // Get dimensions of image block, it will be later used to divide the image into n many image blocks
            let image_block = ImageBlock::new(
                image_block_height!(
                    input_pipeline_slice.data.as_ref().unwrap().len(),
                    image_data_tensor_shape.get_channels()
                ),
                /*image_block_height_vertical!(image_data_tensor_shape),*/
                image_block_width!(
                    input_pipeline_slice.data.as_ref().unwrap().len(),
                    image_data_tensor_shape.get_channels()
                ),
                /*image_block_width_vertical!(image_data_tensor_shape),*/
                image_block_size!(
                    input_pipeline_slice.data.as_ref().unwrap().len(),
                    image_data_tensor_shape.get_channels()
                ), /*image_block_size_vertical!(image_data_tensor_shape) as usize*/
            );

            // Print dimensions of image block
            println!(
                "Height = {}",
                image_block_height!(
                    input_pipeline_slice.data.as_ref().unwrap().len(),
                    image_data_tensor_shape.get_channels()
                )
            );
            println!(
                "Width =  {}",
                image_block_width!(
                    input_pipeline_slice.data.as_ref().unwrap().len(),
                    image_data_tensor_shape.get_channels()
                )
            );
            println!(
                "{}, {}, {}",
                image_block.get_height(),
                image_block.get_width(),
                image_block.get_size()
            );

            /*println! ("Height_Vertical = {}", image_block_width_vertical!(image_data_tensor_shape));
            println! ("Height_Vertical = {}", image_block_height_vertical!(image_data_tensor_shape));*/

            //panic! ("JUST GOT OUT!");

            // Generate random context and target block numbers
            for j in 0..random_context_target_block_numbers.len() {
                let mut random_number: u8;

                loop {
                    random_number =
                        random_whole_number(1, random_context_target_block_numbers.len()) as u8;
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

            // Get dimensions of image block
            let dims_image_block = Box::new(Dimensions::new(
                image_block.get_width(),
                image_block.get_height(),
            ));

            /*
             * CALCULATE NUMBER OF BLOCKS REQUIRED TO COVER IMAGE WIDTH
             *
             * This section computes how many blocks of fixed width are needed to span
             * the entire width of the image, accounting for potential partial coverage
             * at the image boundary.
             *
             * Steps:
             * 1. Compute base number of complete blocks that fit within image width
             * 2. Calculate total pixels covered by these complete blocks
             * 3. If complete blocks don't fully cover the image width (i.e., there are
             *    remaining pixels), add an additional partial block to cover them
             *
             * Example:
             *   Image width: 100px, Block width: 30px
             *   Base blocks: 100/30 = 3 (floor division)
             *   Covered pixels: 3 * 30 = 90px
             *   Remaining pixels: 10px -> Add 1 partial block
             *   Total blocks: 4
             */
            let mut number_of_blocks_per_line: f64 =
                dims_image.get_width() / dims_image_block.get_width();

            let mut overlapping_pixels_per_line = dims_image.get_width()
                - number_of_blocks_per_line.floor() * dims_image_block.get_width();

            if overlapping_pixels_per_line > 0.0 {
                number_of_blocks_per_line = number_of_blocks_per_line.floor();

                number_of_blocks_per_line += 1.0;
            }

            /*
             * CALCULATE NUMBER OF BLOCKS REQUIRED TO COVER IMAGE HEIGHT
             *
             * This section computes how many blocks of fixed height are needed to span
             * the entire height of the image, accounting for potential partial coverage
             * at the image boundary.
             *
             * Steps:
             * 1. Compute base number of complete blocks that fit within image height
             * 2. Calculate total pixels covered by these complete blocks
             * 3. If complete blocks don't fully cover the image height (i.e., there are
             *    remaining pixels), add an additional partial block to cover them
             *
             * Example:
             *   Image height: 100px, Block height: 30px
             *   Base blocks: 100/30 = 3 (floor division)
             *   Covered pixels: 3 * 30 = 90px
             *   Remaining pixels: 10px -> Add 1 partial block
             *   Total blocks: 4
             */
            let mut number_of_blocks_per_column: f64 =
                dims_image.get_height() / dims_image_block.get_height();

            let mut overlapping_pixels_per_column = dims_image.get_height()
                - number_of_blocks_per_column.floor() * dims_image_block.get_height();

            if overlapping_pixels_per_column > 0.0 {
                number_of_blocks_per_column = number_of_blocks_per_column.floor() + 1.0;
            }

            block_file_names.clear();

            // Divide each image into n many image blocks, where n is the total number of context and target blocks
            for j in 0..random_context_target_block_numbers.len() {
                println!(
                    "---->>>>>>>>>>>> {}, {}",
                    image_block_slice_start!(
                        random_context_target_block_numbers[j],
                        image_block.get_width(),
                        image_data_tensor_shape.get_channels()
                    ),
                    image_block_slice_end!(
                        random_context_target_block_numbers[j],
                        image_block.get_width() as usize,
                        image_data_tensor_shape.get_channels()
                    )
                );

                image_block_slice_start_experimental!(
                    random_context_target_block_numbers[j],
                    &dims_image,
                    &dims_image_block
                );

                /*let image_block_slice: Box<Collective<T>> = input_pipeline_slice.get_slice(
                    image_block_slice_start!(
                        random_context_target_block_numbers[j],
                        image_block.get_width(),
                        image_data_tensor_shape.get_channels()
                    ),
                    image_block_slice_end!(
                        random_context_target_block_numbers[j],
                        image_block.get_width(),
                        image_data_tensor_shape.get_channels()
                    ),
                    &dims_image_block,
                    Axis::Rows,
                );*/

                println!(
                    "WAR WAR WAR WAR = {}",
                    zero_based_block_number_horizontal_experimental! {
                    random_context_target_block_numbers[j],
                    number_of_blocks_per_line}
                );

                println!(
                    "GET WIDTH = {}, {}",
                    dims_image_block.get_width(),
                    (overlapping_pixels_per_line / number_of_blocks_per_line).floor()
                );

                let image_block_slice: Box<Collective<T>> = input_pipeline_slice.get_slice(
                    image_block_slice_start_vertical_experimental!(
                        random_context_target_block_numbers[j],
                        number_of_blocks_per_line,
                        overlapping_pixels_per_column,
                        &dims_image_block
                    ),
                    image_block_slice_start_horizontal_experimental!(
                        random_context_target_block_numbers[j],
                        number_of_blocks_per_line,
                        /*image_data_tensor_shape.get_channels(),*/
                        overlapping_pixels_per_line,
                        &dims_image_block
                    ) /*/ image_data_tensor_shape.get_channels()*/ as f64,
                    &dims_image_block,
                    Axis::Rows,
                );

                if image_block_slice.data.is_some() {
                    println!(
                        "-> Loope loop..... looop {}",
                        image_block_slice.data.as_ref().unwrap().len()
                    );
                }

                println!(
                    "Block number = {}",
                    random_context_target_block_numbers[j] - 1
                );

                println!(
                    "($image_dims.get_width() / ($block_dims.get_width())) as usize = {}",
                    ((dims_image.get_width()) / (dims_image_block.get_width())) as usize
                );

                println!(
                    "Over Lapping Pixels Per Column = {}",
                    overlapping_pixels_per_column
                );

                println!(
                    "Number of block per column = {}",
                    number_of_blocks_per_column
                );

                /*input_pipeline_slice.get_slice(
                    image_block_slice_start_vertical_experimental!(
                        random_context_target_block_numbers[j],
                        number_of_blocks_per_line,
                        overlapping_pixels_per_column,
                        &dims_image_block
                    ),
                    image_block_slice_end!(
                        random_context_target_block_numbers[j],
                        image_block.get_width(),
                        image_data_tensor_shape.get_channels()
                    ),
                    &dims_image_block,
                    Axis::Rows,
                );*/

                println!(
                    "IMAGE BLOCK SLICE START VERTICAL = {}",
                    image_block_slice_start_vertical_experimental!(
                        random_context_target_block_numbers[j],
                        number_of_blocks_per_line,
                        overlapping_pixels_per_column,
                        &dims_image_block
                    )
                );

                println!(
                    "IMAGE BLOCK SLICE_END_VERTICAL = {}",
                    image_block_slice_end_vertical_experimental!(
                        random_context_target_block_numbers[j],
                        number_of_blocks_per_line,
                        overlapping_pixels_per_column,
                        &dims_image_block
                    )
                );

                println!(
                    "IMAGE BLOCK SLICE START HORIZONTAL = {}",
                    image_block_slice_start_horizontal_experimental!(
                        random_context_target_block_numbers[j],
                        number_of_blocks_per_line,
                        /*image_data_tensor_shape.get_channels(),*/
                        overlapping_pixels_per_line,
                        &dims_image_block
                    )
                );

                println!(
                    "IMAGE BLOCK SLICE END HORIZONTAL = {}",
                    image_block_slice_end_horizontal_experimental!(
                        random_context_target_block_numbers[j],
                        number_of_blocks_per_line,
                        image_data_tensor_shape.get_channels(),
                        overlapping_pixels_per_line,
                        &dims_image_block
                    )
                );

                /*let block_line_number = image_block_slice_start_vertical_experimental!(
                    random_context_target_block_numbers[j],
                    &dims_image,
                    &dims_image_block
                );*/

                /*if image_block_slice.data.is_some() {
                    println!(
                        "-> Loope loop..... looop {}",
                        image_block_slice.data.as_ref().unwrap().len()
                    );
                }*/

                //let path_text = format!("{}{}{}{}{}", JEPA_IMAGE_BLOCK_FILE_NAME_PRELUDE, random_context_target_block_numbers[j], JEPA_IMAGE_BLOCK_FILE_NAME_POSTLUDE, i + 1, JEPA_IMAGE_BLOCK_FILE_NAME_EXTENSION);
                path_text = format!(
                    "{}{}{}{}{}",
                    JEPA_IMAGE_BLOCK_FILE_NAME_PRELUDE,
                    random_context_target_block_numbers[j],
                    JEPA_IMAGE_BLOCK_FILE_NAME_POSTLUDE,
                    i + 1,
                    PNG_FILE_EXTENSION
                );

                block_file_names.push(path_text.clone());

                path = Path::new(&path_text);
                //println! ("{}", path.display());

                png_png = create_png_from_collective::<T>(&image_block_slice, &path);

                match png_png {
                    Some(png) => {
                        png.traverse();

                        //println!("Saving PNG file: {}", output_path.display());
                        //png.save_to_file(&output_path);

                        png.save_to_file(&path);
                    }
                    None => {
                        println!("Failed to create PNG from boxed deflated data");
                    }
                }
            }

            /*for j in 0..random_context_target_block_numbers.len() {

                println! ("-> {}", random_context_target_block_numbers[j]);
            }*/

            // Now we have the random context and target block numbers
            // We can use them to create the context and target encoders

            //println!("-> {}", (input_pipeline_slice.data.as_ref().unwrap().len()/image_data_tensor_shape.get_channels()));
            //println!("-> {}", ((input_pipeline_slice.data.as_ref().unwrap().len()/image_data_tensor_shape.get_channels())/8) as f64);

            //let image_block = ImageBlock::new(image_data_tensor_shape.get_height() as f64, image_data_tensor_shape.get_width() as f64, (input_pipeline_slice.data.as_ref().unwrap().len()/image_data_tensor_shape.get_channels())/8);

            //println! ("-> H = {}", (((input_pipeline_slice.data.as_ref().unwrap().len()/image_data_tensor_shape.get_channels())/(JEPA_NUMBER_OF_CONTEXT_BLOCKS + JEPA_NUMBER_OF_TARGET_BLOCKS)) as f64 / JEPA_IMAGES_ASPECT_RATIO).sqrt());
            //println! ("-> W = {}", (((input_pipeline_slice.data.as_ref().unwrap().len()/image_data_tensor_shape.get_channels())/(JEPA_NUMBER_OF_CONTEXT_BLOCKS + JEPA_NUMBER_OF_TARGET_BLOCKS)) as f64 / JEPA_IMAGES_ASPECT_RATIO).sqrt() *//  JEPA_IMAGES_ASPECT_RATIO);

            /*
                block_file_names.sort();
                create_png_from_png_files::<T> (&block_file_names, 254, 344, 3);
            */
        }

        println!("ENDING HERE HERE");
    }
}
