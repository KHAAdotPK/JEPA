/*
 * implementation/sundry.rs
 * Q@khaa.pk
 */

use rand::Rng;

pub fn random_whole_number(min: u32, max: u32) -> u32 {

    let mut rng = rand::thread_rng();
    rng.gen_range(min..=max)
}