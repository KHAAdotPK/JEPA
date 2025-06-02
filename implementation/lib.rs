/*
 * implementation/lib.rs
 * Q@khaa.pk
 */

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

/// Represents the dimensional structure of image data tensors.
/// 
/// This struct defines the shape characteristics of image data used in machine learning
/// models, particularly for computer vision tasks. It encapsulates the three fundamental
/// dimensions that define image data structure.
/// 
/// # Fields
/// * `height` - The vertical dimension of the image in pixels
/// * `width` - The horizontal dimension of the image in pixels  
/// * `channels` - The number of color channels (e.g., 3 for RGB, 1 for grayscale)
/// 
/// # Example
/// ```rust
/// // RGB image of 224x224 pixels
/// let shape = ImageDataTensorShape::new(224, 224, 3);
/// 
/// // Grayscale image of 28x28 pixels (like MNIST)
/// let mnist_shape = ImageDataTensorShape::new(28, 28, 1);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ImageDataTensorShape {

    height: usize,
    width: usize,
    channels: usize,
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

    pub fn new(height: usize, width: usize, channels: usize) -> ImageDataTensorShape {

        ImageDataTensorShape {

            height: height,
            width: width,
            channels: channels,
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

    pub fn get_ModelConfig(&self) -> ModelConfig {

        self.model_config
    }

    pub fn get_ImageDataTensorShape(&self) -> ImageDataTensorShape {

        self.image_data_tensor_shape
    }
    
    pub fn start_training_loop(&self) {
        
    }
}
