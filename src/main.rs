/*
 * src/main.rs
 * Q@khaa.pk
 */

#![allow(non_snake_case)]

#[path = "../lib/numrs/mod.rs"]
mod numrs;

use std::{cell::RefCell, fs::{File, metadata}, io::Read, path::Path, rc::Rc, str};
use argsv::{start, find_arg, stop, help, help_line, common_argc, process_argument};
use numrs::{dimensions::Dimensions, collective::Collective, num::Numrs};
use png::{Png, Chunk, InflatedData}; 

/*#[link(name = "png", kind = "dylib")]
extern {
 
    fn big_endian_read_u32(ptr: *const u8) -> u32;     
}*/

fn main() {

    let command_lines = "h -h help --help ? /? (Displays help screen)\n\
                         v -v version --version /v (Displays version number)\n\
                         t -t traverse --traverse /t (Traverses PNG file structure and displays it)\n\
                         d -d delete --delete /d (Deletes the named chunk from the PNG file)\r\n\
                         verbose --verbose (Displays detailed information or feedback about the execution of another command)";

    // Get the command-line arguments as an iterator
    let args: Vec<String> = std::env::args().collect();
    // Create a Vec<CString> from the Vec<String>
    let c_args: Vec<std::ffi::CString> = args.iter().map(|s| std::ffi::CString::new(s.as_str()).expect("Failed to create CString")).collect();
    // Get the equivalent of argv in C. Create a Vec<*const c_char> from the Vec<CString>.
    let c_argv: Vec<*const std::os::raw::c_char> = c_args.iter().map(|s| s.as_ptr()).collect();
    // Get the C equivalent of argc.    
    let argc: i32 = c_args.len() as std::os::raw::c_int;

    let mut ncommon: i32 = 0;

    let head = start (argc, c_argv, command_lines);
        
        ncommon = common_argc (head);

        let arg_help = find_arg (head, command_lines, "h");
        if !arg_help.is_null() || argc < 1 {
            
            help (head, command_lines);
            stop (head); 

            return;
        }

    stop (head); 
           
    // for loop with range
    for i in 1..ncommon {

        let arg = &args[i as usize];

        let path: &Path = Path::new(arg);

        // Check if the file exists and has a PNG extension
        if path.exists() && path.extension().map_or(false, |ext| ext == "png") {

            println!("Processing PNG file: {}", arg);
        
            // Here you would add code to read/process the PNG file
            // For example:
            // let image = image::open(path).expect("Failed to open image");
            // Or any other PNG processing logic you need

            /*
                The file will be closed once the scope of its owner ends. 
                If you need it to live for less time, you can introduce a new scope where it will live.
                If you need it to live for more time, you can move the ownership of the file to a new owner.
            */
            let file = File::open(&path);
            let mut buffer: Vec<u8> = Vec::new();

            match file {

                Err (why) => {
        
                    panic!("Couldn't open {}: {}", path.display().to_string(), why);    
                }
        
                Ok (mut f) => {
        
                    buffer = vec![0; metadata(arg).unwrap().len() as usize];
        
                    f.read (&mut buffer).unwrap();
        
                    /*    
                        The idiomatic way to control how long it's open is to use a scope { }.
                        The file will be automatically dropped when the "scope" is done (this is usually when a function exits).
                        There's one other way to manually close the file, using the drop() function. The drop() function does the exact same thing as what happens when the scope around the file closes. 
                     */
                    drop(f); 
                    
                    let png = Png::new(buffer);
                    let mut iter = png.chunks.iter();

                    // Create a buffer to hold all concatenated IDAT data
                    let mut all_idat_data = Vec::new();

                    println!("Number of chunks = {}", png.chunks.len());

                    while let Some(chunk) = iter.next() {
                        //println!("Length = {}", unsafe { big_endian_read_u32(chunk.length.clone().as_mut_ptr()) });
                        println!("Length = {}", chunk.get_length() );
                        println!("Type = [ {} {} {} {} ], {}", 
                            chunk.type_name[0], 
                            chunk.type_name[1], 
                            chunk.type_name[2], 
                            chunk.type_name[3],                             
                            chunk.get_type_name()
                        );

                        if chunk.get_type_name() == "IDAT" {
                            
                            // Check if it matches the actual data length
                            assert_eq!(chunk.get_length() as usize, chunk.data.len());

                            all_idat_data.extend_from_slice(&chunk.data);

                            //let dat = chunk.get_inflated_data();
                        }
                    }

                    println!("Length of all_idat_data = {}", all_idat_data.len());

                    let dat: *mut InflatedData = png.get_inflated_data(&all_idat_data);
                }
            }

        } else {
            println!("Invalid or non-existent PNG file: {}", arg);
        }                    
    }
} 
