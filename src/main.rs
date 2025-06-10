/*
 * src/main.rs
 * Q@khaa.pk
 */

#![allow(non_snake_case)]

use std::{cell::RefCell, fs::{File, metadata}, io::Read, io::Write, path::Path, path::PathBuf, rc::Rc, str};
use argsv::{common_argc, find_arg, help, help_line, process_argument, start, stop, COMMANDLINES, PCLA};
use numrs::{dimensions::Dimensions, collective::Collective, num::Numrs};
use png::{constants, Png, Chunk, DeflatedData, InflatedData, create_uncompressed_png, modify_png_pixel_data}; 

use jepa::images::{Model, ModelConfig, ImageDataTensorShape, ImageDataTensorShapeFormat};

fn main() {

    let command_lines = "h -h help --help ? /? (Displays help screen)\n\
                         v -v version --version /v (Displays version number)\n\
                         t -t traverse --traverse /t (Traverses PNG file structure and displays it)\n\
                         d -d delete --delete /d (Deletes the named chunk from the PNG file)\n\
                         verbose --verbose (Displays detailed information or feedback about the execution of another command)\n\
                         suffix --suffix (Suffix for the output PNG file)\n";

    let arg_suffix: *mut COMMANDLINES;

    let mut suffix_token: Option<&str> = Some(constants::PNG_OUTPUT_FILE_SUFFIX);                 

    // Get the command-line arguments as an iterator
    let args: Vec<String> = std::env::args().collect();
    // Create a Vec<CString> from the Vec<String>
    let c_args: Vec<std::ffi::CString> = args.iter().map(|s| std::ffi::CString::new(s.as_str()).expect("Failed to create CString")).collect();
    // Get the equivalent of argv in C. Create a Vec<*const c_char> from the Vec<CString>.
    let c_argv: Vec<*const std::os::raw::c_char> = c_args.iter().map(|s| s.as_ptr()).collect();
    // Get the C equivalent of argc.    
    let argc: i32 = c_args.len() as std::os::raw::c_int;

    let mut ncommon: i32 = 0;

    let head = start (argc, c_argv.clone(), command_lines);
        
        ncommon = common_argc (head);

        let arg_help = find_arg (head, command_lines, "h");
        if !arg_help.is_null() || argc < 1 {
            
            help (head, command_lines);
            stop (head); 

            return;
        }
        
        arg_suffix = find_arg (head, command_lines, "--suffix");
        if !arg_suffix.is_null() {

            let arg_suffix_clap: *mut PCLA = unsafe {(*arg_suffix).get_clap()};
            if unsafe{(*arg_suffix_clap).get_argc()} > 0 {

                suffix_token = Some(unsafe { str::from_utf8_unchecked(&args[unsafe{(*arg_suffix_clap).get_index_number() as usize} + 1].as_bytes()) });
             } 
        } 

        /*match suffix_token {

            Some (suffix) => {

                println!("Suffix token: {}", suffix);
            }
            None => {

                println!("No suffix token provided, using default.");
            }
        }*/

    stop (head); 
           
    // for loop with range
    for i in 1..ncommon {

        let arg = &args[i as usize];

        let path: &Path = Path::new(arg);

        let mut height: u32 = 0;
        let mut width: u32 = 0;

        let mut color_type: u8 = 0;
        let mut bit_depth: u8 = 0;

        // Check if the file exists and has a PNG extension
        if path.exists() && path.extension().map_or(false, |ext| ext == "png") {

            println!("Processing PNG file: {}", arg);
                   
            /*
                The file will be closed once the scope of its owner ends. 
                If you need it to live for less time, you can introduce a new scope where it will live.
                If you need it to live for more time, you can move the ownership of the file to a new owner.
            */
            let file = File::open(&path);
            
            match file {

                Err (why) => {
        
                    //panic!("Couldn't open {}: {}", path.display().to_string(), why);

                    println!("Skipping file: {}, couldn't open because of {}.", path.display().to_string(), why);
                    continue; // Skip to next file in the loop
                }
                                        
                Ok (mut f) => {
                            
                    let mut buffer = vec![0; metadata(arg).unwrap().len() as usize];
        
                    f.read (&mut buffer).unwrap();
        
                    /*    
                        The idiomatic way to control how long it's open is to use a scope { }.
                        The file will be automatically dropped when the "scope" is done (this is usually when a function exits).
                        There's one other way to manually close the file, using the drop() function. The drop() function does the exact same thing as what happens when the scope around the file closes. 
                     */
                    drop(f); 
                    
                    let png = Png::new(buffer);

                    match png.match_color_type_and_bit_depth(2, 8) {
                                                
                        false => {

                            println!("Skipping file: {}, it has unsupported color type/bit depth combination.", path.display().to_string());
                            continue; // Skip to next file in the loop    
                        },
                        _ => {

                        }                        
                    }

                    let chunk: Option<&Chunk> = png.get_chunk_by_type("IHDR");

                    match chunk {

                        Some (chunk) => {

                            height = chunk.get_height();
                            width = chunk.get_width();

                            color_type = chunk.get_color_type();
                            bit_depth = chunk.get_bit_depth();
                        }
                        None => {

                        }
                    }

                    //let all_idat_data_deflated: *mut DeflatedData = png.get_all_idat_data_as_DeflatedData();

                    let all_idat_data: Vec<u8> = png.get_all_idat_data_as_vec();
                    let dat: *mut InflatedData = png.get_inflated_data(&all_idat_data);

                    /* TESTING FOR RGB BEGINGS HERE */

                        let dat_modify = modify_png_pixel_data(dat, Vec::from([0xFF, 0x00, 0x00]), width, height, color_type, bit_depth);
                                                                   
                    /* TESTING FOR TGB ENDS HERE */

                    let deflated_data: *mut DeflatedData = png.get_deflated_data(dat_modify);
                                        
                    let output_path = path.with_extension("").with_extension(&format!("{}.png", suffix_token.unwrap()));

                    println!("Output PNG file will be: {}", output_path.display());
                    
                    create_uncompressed_png(width, height, deflated_data as *mut InflatedData, &output_path);
                }
            }

        } else {

            println!("Invalid or non-existent PNG file: {}", arg);
        }                    
    }
} 
