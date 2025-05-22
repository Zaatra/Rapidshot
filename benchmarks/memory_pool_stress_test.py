import time
import argparse
import logging
import rapidshot # Assuming rapidshot.create is the entry point

# Setup basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_region_arg(region_str):
    """Parses a region string 'left,top,right,bottom' into a tuple of integers."""
    if not region_str:
        return None
    try:
        parts = list(map(int, region_str.split(',')))
        if len(parts) == 4:
            # Basic validation: left < right, top < bottom
            if not (parts[0] < parts[2] and parts[1] < parts[3]):
                raise argparse.ArgumentTypeError(
                    f"Invalid region '{region_str}': left must be < right and top must be < bottom."
                )
            return tuple(parts)
        else:
            raise argparse.ArgumentTypeError("Region must be 'left,top,right,bottom'.")
    except ValueError:
        raise argparse.ArgumentTypeError("Region components must be integers.")
    except Exception as e: # Catch other validation errors
        raise argparse.ArgumentTypeError(str(e))


def main():
    parser = argparse.ArgumentParser(description="Memory Pool Stress Test for RapidShot's grab() method.")
    parser.add_argument(
        "--num_grabs",
        type=int,
        default=10000,
        help="Number of grab-release cycles to perform."
    )
    parser.add_argument(
        "--processor",
        type=str,
        choices=["numpy", "cupy"],
        default="numpy",
        help="Processing backend to use ('numpy' or 'cupy')."
    )
    parser.add_argument(
        "--region",
        type=parse_region_arg,
        default=None,
        help="Optional capture region as 'left,top,right,bottom'. Full screen if not specified."
    )
    parser.add_argument(
        "--pool_size",
        type=int,
        default=5, # A small pool size can stress the checkout/checkin more.
        help="Number of buffers in the ScreenCapture memory pool."
    )
    parser.add_argument(
        "--display_idx",
        type=int,
        default=0,
        help="Index of the display/monitor to capture from."
    )
    parser.add_argument(
        "--output_idx",
        type=int,
        default=0,
        help="Index of the output on the selected display."
    )


    args = parser.parse_args()

    logger.info(
        f"Starting Memory Pool Stress Test with: "
        f"NumGrabs={args.num_grabs}, Processor={args.processor}, "
        f"Region={args.region if args.region else 'Default (Full Screen)'}, PoolSize={args.pool_size}"
    )

    # Initialization
    try:
        # Assuming rapidshot.create is the simplified way to get a ScreenCapture instance
        # The 'create' function should internally handle device/output selection and processor init.
        screencapture = rapidshot.create(
            display_idx=args.display_idx,
            output_idx=args.output_idx,
            use_nvidia_gpu=(args.processor == 'cupy'),
            pool_size_frames=args.pool_size,
            # region is passed to grab, Screencapture initializes with full screen or its own default
        )
        logger.info(f"ScreenCapture instance created with processor: {args.processor}, pool size: {args.pool_size}")
        if args.region:
             logger.info(f"Using specified region for grab: {args.region}")
        else:
            logger.info(f"Using default region (likely full screen) for grab, configured by ScreenCapture pool: {screencapture.region}")


    except Exception as e:
        logger.error(f"Error initializing ScreenCapture: {e}")
        return

    failed_grabs = 0
    region_to_grab = args.region # This can be None for default, or a specific tuple

    # Benchmarking Loop
    logger.info(f"Starting benchmarking loop for {args.num_grabs} grab-release cycles...")
    t_start = time.perf_counter()

    for i in range(args.num_grabs):
        frame_or_wrapper = screencapture.grab(region=region_to_grab)
        
        if frame_or_wrapper is not None:
            if hasattr(frame_or_wrapper, 'release'): # Check if it's a PooledBuffer
                try:
                    frame_or_wrapper.release()
                except Exception as e:
                    logger.warning(f"Error releasing PooledBuffer on iteration {i}: {e}")
                    failed_grabs +=1 # Consider this a failure if release fails
            # Else: it's a raw array (pool bypassed or shape changed), no release needed here.
        else:
            # grab() returned None, indicating a failure (e.g., pool exhausted, DDA error)
            if (i % (args.num_grabs // 100 if args.num_grabs >=100 else 100) == 0) or args.num_grabs < 100 : # Log periodically or if few grabs
                 logger.warning(f"Grab failed on iteration {i+1}. Pool might be temporarily exhausted or error occurred.")
            failed_grabs += 1
            # Small sleep if grabs are failing, to avoid spamming logs or tight loop on errors
            if failed_grabs > args.num_grabs * 0.1 and failed_grabs % 10 == 0: # If many failures
                time.sleep(0.001)


    t_end = time.perf_counter()
    logger.info("Benchmarking loop finished.")

    # Results
    total_time = t_end - t_start
    successful_grabs = args.num_grabs - failed_grabs
    grabs_per_second = 0
    if total_time > 0 and successful_grabs > 0 : # Avoid division by zero
        grabs_per_second = successful_grabs / total_time

    logger.info("--- Benchmark Results ---")
    logger.info(f"Total Grabs Attempted: {args.num_grabs}")
    logger.info(f"Successful Grabs:      {successful_grabs}")
    logger.info(f"Failed Grabs:          {failed_grabs}")
    logger.info(f"Total Time Taken:      {total_time:.4f} seconds")
    if successful_grabs > 0 :
        logger.info(f"Grabs Per Second (Successful): {grabs_per_second:.2f} FPS")
    else:
        logger.info("Grabs Per Second (Successful): N/A (no successful grabs)")


    # Cleanup
    logger.info("Releasing ScreenCapture resources.")
    try:
        screencapture.release()
    except Exception as e:
        logger.error(f"Error during ScreenCapture release: {e}")
    
    logger.info("Benchmark finished.")

if __name__ == "__main__":
    main()
