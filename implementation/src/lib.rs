/*
    implementation/src/lib.rs
    Q@khaa.pk
 */

/*use std::{rc::Rc, cell::RefCell}; 
use numrs::{dimensions::Dimensions, collective::Collective, num::Numrs};*/

// JEPA framework for images 
pub mod images;

/*use images::ImageDataTensorShape;*/

/*
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
}*/
