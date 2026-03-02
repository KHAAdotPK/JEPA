/*
 * implementation/src/sundry.rs
 * Q@khaa.pk
 */

#![allow(dead_code)]

pub const JEPA_IMAGE_BLOCK_FILE_NAME_PRELUDE: &str = "jepa_image_block_";
pub const JEPA_IMAGE_BLOCK_FILE_NAME_POSTLUDE: &str = "_of_image_";
pub const JEPA_IMAGE_BLOCK_FILE_NAME_EXTENSION: &str = ".png";

pub const JEPA_IMAGE_HEIGHT: f64 = 344.0;
pub const JEPA_IMAGE_WIDTH: f64 = 254.0;
pub const JEPA_IMAGE_BLOCK_WIDTH: f64 = JEPA_IMAGE_WIDTH / 2.0;
pub const JEPA_IMAGE_BLOCK_HEIGHT: f64 = JEPA_IMAGE_HEIGHT / 4.0;
pub const JEPA_IMAGE_CHANNELS: usize = 3;
pub const JEPA_IMAGE_COLOR_TYPE: u8 = 2;
pub const JEPA_IMAGE_BIT_DEPTH: u8 = 8;
pub const JEPA_IMAGE_SIZE: usize =
    (JEPA_IMAGE_HEIGHT * JEPA_IMAGE_WIDTH * JEPA_IMAGE_CHANNELS as f64) as usize;
pub const JEPA_NUMBER_OF_CONTEXT_BLOCKS: usize = 4;
pub const JEPA_NUMBER_OF_TARGET_BLOCKS: usize = 4;
pub const JEPA_IMAGES_ASPECT_RATIO: f64 = 0.75;
