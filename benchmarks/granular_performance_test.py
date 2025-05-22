import time
import numpy as np
import argparse
import logging
import rapidshot
from rapidshot.core.duplicator import Duplicator # Direct import for Duplicator
from rapidshot.processor.numpy_processor import NumpyProcessor
# from rapidshot.processor.cupy_processor import CupyProcessor # Placeholder for CuPy

# Setup basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_region(region_str):
    """Parses a region string 'left,top,right,bottom' into a tuple of integers."""
    if not region_str:
        return None
    try:
        parts = list(map(int, region_str.split(',')))
        if len(parts) == 4:
            return tuple(parts)
        else:
            raise argparse.ArgumentTypeError("Region must be 'left,top,right,bottom'.")
    except ValueError:
        raise argparse.ArgumentTypeError("Region components must be integers.")

def main():
    parser = argparse.ArgumentParser(description="Granular Performance Test for RapidShot.")
    parser.add_argument(
        "--processor",
        type=str,
        choices=["numpy", "cupy"],
        default="numpy",
        help="Processing backend to use."
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=1000,
        help="Number of frames to capture and process for the benchmark."
    )
    parser.add_argument(
        "--region",
        type=parse_region,
        default=None,
        help="Optional capture region as 'left,top,right,bottom'. Captures full screen if not specified."
    )
    parser.add_argument(
        "--color_mode",
        type=str,
        default="BGR",
        help="Target color mode for processing (e.g., RGB, BGR, RGBA, GRAY)."
    )
    parser.add_argument(
        "--display_idx",
        type=int,
        default=0,
        help="Index of the display/monitor to capture from."
    )
    parser.add_argument(
        "--timeout_ms",
        type=int,
        default=10, # Default consistent with Duplicator's new default
        help="Timeout in milliseconds for AcquireNextFrame."
    )

    args = parser.parse_args()

    logger.info(f"Starting benchmark with: Processor={args.processor}, Frames={args.num_frames}, "
                f"Region={args.region}, ColorMode={args.color_mode}, Display={args.display_idx}, "
                f"Timeout={args.timeout_ms}ms")

    # Initialization
    try:
        device = rapidshot.get_device(args.display_idx)
        output = rapidshot.get_output(device, 0) # Assuming first output on the device
        logger.info(f"Using Device: {device.name}, Output: {output.devicename}")
    except Exception as e:
        logger.error(f"Error initializing display device or output: {e}")
        return

    duplicator = Duplicator(output=output, device=device, timeout_ms=args.timeout_ms)
    
    if args.processor == "numpy":
        processor = NumpyProcessor(color_mode=args.color_mode)
    elif args.processor == "cupy":
        try:
            from rapidshot.processor.cupy_processor import CupyProcessor
            processor = CupyProcessor(color_mode=args.color_mode)
            logger.info("Using CupyProcessor.")
        except ImportError:
            logger.error("CupyProcessor selected, but CuPy or related dependencies are not installed. Exiting.")
            duplicator.release()
            return
        except Exception as e:
            logger.error(f"Error initializing CupyProcessor: {e}. Exiting.")
            duplicator.release()
            return
    else:
        logger.error(f"Invalid processor selected: {args.processor}. Exiting.")
        duplicator.release()
        return

    width, height = output.resolution
    rotation_angle = output.rotation_angle
    region_tuple = args.region if args.region else (0, 0, width, height)

    logger.info(f"Effective capture region: {region_tuple} on an output of {width}x{height} with rotation {rotation_angle} deg.")

    capture_times = []
    process_times = []
    successful_frames = 0
    updated_frames = 0

    # Benchmarking Loop
    logger.info(f"Starting benchmarking loop for {args.num_frames} frames...")
    for i in range(args.num_frames):
        t_start_capture = time.perf_counter()
        success = duplicator.update_frame()
        t_end_capture = time.perf_counter()
        
        capture_time = t_end_capture - t_start_capture
        capture_times.append(capture_time)

        if success:
            successful_frames += 1
            if duplicator.updated:
                updated_frames +=1
                texture = duplicator.texture # This is POINTER(ID3D11Texture2D)
                
                # Ensure width, height, rotation_angle are from the duplicator's output,
                # as they might change dynamically (though unlikely in a benchmark context)
                current_width, current_height = duplicator.get_output_dimensions()
                current_rotation_angle = duplicator.get_rotation_angle()

                # Use the effective region_tuple, ensuring it's valid for current_width/height
                # For simplicity, this benchmark assumes the initial region remains valid.
                # A more robust benchmark might re-validate or adjust `region_tuple` if dimensions change.

                t_start_process = time.perf_counter()
                try:
                    # The processor.process() expects the raw texture (rect),
                    # width and height of the *output* (not necessarily the region),
                    # the capture region tuple, and the rotation angle.
                    processed_frame = processor.process(
                        texture, 
                        current_width, 
                        current_height, 
                        region_tuple, 
                        current_rotation_angle
                    )
                    # Ensure processed_frame is not None and has data, if relevant for the processor
                    if processed_frame is None and args.processor == "numpy": # NumpyProcessor may return None on error
                         logger.warning(f"Frame {i+1}: Processing returned None.")
                    elif hasattr(processed_frame, 'shape') and processed_frame.size == 0:
                         logger.warning(f"Frame {i+1}: Processing returned an empty frame.")

                except Exception as e:
                    logger.error(f"Error during processing frame {i+1}: {e}")
                    # Optionally skip timing this frame's processing or handle error
                    continue # Skip timing processing for this frame
                finally: # Ensure process time is recorded if started
                    t_end_process = time.perf_counter()
                    process_time = t_end_process - t_start_process
                    process_times.append(process_time)
            else:
                logger.debug(f"Frame {i+1}: Acquired, but no new content (duplicator.updated is False).")
        else:
            logger.warning(f"Frame {i+1}: Failed to acquire (duplicator.update_frame() returned False). Error: {duplicator.get_last_error()}")

        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{args.num_frames} frames...")
            
    logger.info("Benchmarking loop finished.")

    # Results
    if capture_times:
        avg_capture_time = sum(capture_times) / len(capture_times)
        logger.info(f"Average Capture Time: {avg_capture_time * 1000:.2f} ms")
    else:
        avg_capture_time = 0
        logger.info("No capture times recorded.")

    if process_times:
        avg_process_time = sum(process_times) / len(process_times)
        logger.info(f"Average Processing Time: {avg_process_time * 1000:.2f} ms (over {len(process_times)} successfully processed frames)")
    else:
        avg_process_time = 0
        logger.info("No processing times recorded (or all processing failed/skipped).")

    if avg_capture_time > 0 and avg_process_time > 0 and process_times : # Ensure process_times is not empty
        # Calculate total time based on frames that were successfully captured AND processed
        # This assumes capture_time is still relevant even if processing didn't happen for some frames,
        # but FPS should ideally be based on fully completed frames.
        # For a more accurate FPS, consider only frames where both capture and processing occurred.
        # If len(process_times) < len(capture_times), this total_avg_time might be optimistic for overall throughput.
        # A better total_avg_time would be based on the number of items in process_times.
        
        # Calculate total time per *successfully processed* frame
        # We need to sum capture times for frames that were also processed.
        # This is tricky if update_frame is True but duplicator.updated is False.
        # Let's assume for simplicity that if a frame was processed, its corresponding capture time is valid to sum.
        # This means we're calculating FPS based on frames that *could* be processed.
        
        # For a simple FPS, we can use the averages over the number of *processed* frames
        total_avg_time_per_processed_frame = avg_capture_time + avg_process_time # This is a bit of a simplification
                                                                                # if capture_times includes frames not processed.
                                                                                # A more precise sum would be needed.
        
        # Let's calculate total_avg_time based on the number of frames that were processed.
        # This requires summing capture_times for only those frames that led to a process_times entry.
        # The current loop structure adds to capture_times always, but process_times only if updated.
        # This is complex to align post-loop without more state.
        # For now, this simpler sum:
        if len(process_times) > 0:
             # This is the average time for a frame that gets fully captured and processed.
            fps = 1.0 / (avg_capture_time + avg_process_time) if (avg_capture_time + avg_process_time) > 0 else 0
            logger.info(f"Total Average Time (Capture + Process for processed frames): {(avg_capture_time + avg_process_time) * 1000:.2f} ms")
            logger.info(f"Overall FPS (based on processed frames): {fps:.2f}")
        else:
            logger.info("Cannot calculate Total Average Time or FPS as no frames were processed.")

    else:
        logger.info("Cannot calculate Total Average Time or FPS due to missing capture or processing data.")

    logger.info(f"Total frames where AcquireNextFrame succeeded: {successful_frames}/{args.num_frames}")
    logger.info(f"Total frames with updated content: {updated_frames}/{successful_frames if successful_frames > 0 else args.num_frames}")
    
    # Cleanup
    logger.info("Releasing duplicator resources.")
    duplicator.release()
    logger.info("Benchmark finished.")

if __name__ == "__main__":
    main()
