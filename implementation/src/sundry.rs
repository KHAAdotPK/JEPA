/*
 * implementation/src/sundry.rs
 * Q@khaa.pk
 */

use rand::Rng;

pub fn random_whole_number(min: usize, max: usize) -> usize {

    let mut rng = rand::thread_rng();
    rng.gen_range(min..=max)
}