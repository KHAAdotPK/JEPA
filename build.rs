//use std::env;

fn main() {

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set by Cargo");



    println!("cargo:warning=OUT_DIR is {}", out_dir);    
}