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
use png::{Png, Chunk, DeflatedData, InflatedData}; 

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
                        }
                    }

                    println!("Length of all_idat_data = {}", all_idat_data.len());

                    let dat: *mut InflatedData = png.get_inflated_data(&all_idat_data);

                    /*                        
                        This Rust code snippet demonstrates different ways to access the len() method of a DeflatedData struct when you have a raw pointer (*mut DeflatedData) to it.
                        In Rust, raw pointers (*const T or *mut T) are a way to work with memory directly, similar to pointers in C/C++.
                        However, they bypass many of Rust's safety guarantees. One such limitation is that you cannot directly call methods (like len()) on a raw pointer as if it were a reference.
                        Methods are typically defined on the type itself or its references (&T or &mut T).

                        The code shows three common options to safely interact with the data pointed to by the raw pointer dat and call its len() method:
                     */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    /*                        
                        The len() method is not available on raw pointers, it is only available on references, in other words DeflatedData::len() is a method on &DeflatedData (a reference).                        
                        Option 1: Use unsafe to dereference the pointer
                        Option 2: Use as_ref() which gives you a reference option
                        Option 3: Use Box::from_raw() to convert the raw pointer to a Box                        
                    */
                    /*                    
                        Option 1: (Commented out): Use unsafe to dereference the pointer. This is the most direct way.
                                  The *dat dereferences the raw pointer dat to get the actual DeflatedData value.
                                  Because dereferencing a raw pointer is an unsafe operation (as the pointer could be null, dangling, or unaligned),
                                  it must be enclosed in an unsafe block. Once dereferenced, (*dat).len() calls the len() method on the DeflatedData instance.
                        Caveat: This is unsafe because if dat is a null pointer or points to invalid memory, dereferencing it will lead to undefined behavior (likely a crash).
                     */
                    //unsafe { println!("Length of dat = {}", (*dat).len()) };
                    
                    /*
                        Option 2: Use as_ref() which gives you a reference Option
                        Purpose: The as_ref() method (called within an unsafe block because it still involves dereferencing a raw pointer) attempts to convert the raw pointer *mut DeflatedData into an Option<&DeflatedData>.
                        - If dat is a non-null pointer, dat.as_ref() returns Some(&DeflatedData), giving a safe reference data_ref to the DeflatedData.
                          You can then call data_ref.len() safely.
                        - If dat is a null pointer, it returns None, and the else block is executed, preventing a null pointer dereference.
                        Benefit: This is safer than direct dereferencing because it explicitly checks for nullness.
                     */
                    /*if let Some(data_ref) = unsafe { dat.as_ref() } {
                        println!("Length of dat = {}", data_ref.len());
                    } else {
                        println!("Data pointer is null");
                    }*/

                    // Option 3: Use Box::from_raw() to convert the raw pointer to a Box
                    // Memory cleanup:
                    // If you're done with dat completely, convert it to a Box
                    // (this will automatically call drop when it goes out of scope)
                    /*
                        Option 3: Use Box::from_raw() to convert the raw pointer to a Box 
                        Purpose: Box::from_raw(dat) takes a raw pointer dat (that was presumably allocated by Rust's allocator, often via Box::into_raw) and converts it back into a Box<DeflatedData>. 
                        - A Box is a smart pointer that owns the data on the heap. Once the data is in a Box, you can call methods on it directly (e.g., boxed_dat.len()).
                        - Memory Management: Crucially, Box::from_raw takes ownership of the memory. When boxed_dat goes out of scope at the end of the unsafe block,
                          Rust's memory management will automatically call the drop implementation for DeflatedData (if any) and deallocate the memory. This is a way to safely manage the lifetime and cleanup of memory that was previously managed by a raw pointer.
                        Caveat: This option should only be used if dat was originally created from a Box using Box::into_raw and you intend to give ownership back to a Box for proper deallocation. Using it on a pointer not allocated this way can lead to memory corruption or double frees. The comment "Memory cleanup" highlights this aspect.
                    */
                    unsafe { 
                        let boxed_dat = Box::from_raw(dat); 

                        println!("Length of boxed_dat = {}", boxed_dat.len());

                        /*
                            Memory cleanup:
                            If you're done with boxed_dat completely, you can drop it explicitly:
                            drop(boxed_dat);
                            This will automatically call the drop implementation for DeflatedData and deallocate the memory.
                        */
                        //drop(boxed_dat); // Commented out because it is implicitly dropped when the scope ends
                    };
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////                   
                }
            }

        } else {
            println!("Invalid or non-existent PNG file: {}", arg);
        }                    
    }
} 
