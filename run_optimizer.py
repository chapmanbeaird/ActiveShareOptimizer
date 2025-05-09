#!/usr/bin/env python3
"""
Command-line entry point for the Active Share Optimizer.
"""

import argparse
from optimizer import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Active Share Portfolio Optimizer")
    
    parser.add_argument("--data-file", default="inputs/optimizer_input_file.xlsm",
                      help="Path to the input data file (default: inputs/optimizer_input_file.xlsm)")
    parser.add_argument("--num-positions", type=int, default=60,
                      help="Exact number of positions required in the portfolio (default: 60)")
    parser.add_argument("--target-active-share", type=float, default=0.55,
                      help="Target Active Share percentage as decimal (default: 0.55)")
    parser.add_argument("--sector-tolerance", type=float, default=0.03,
                      help="Maximum allowed deviation from benchmark sector weights as decimal (default: 0.03)")
    parser.add_argument("--min-position", type=float, default=1.0,
                      help="Minimum position size as percentage (default: 1.0)")
    parser.add_argument("--max-position", type=float, default=5.0,
                      help="Maximum position size as percentage (default: 5.0)")
    parser.add_argument("--core-rank-limit", type=int, default=3,
                      help="Only consider stocks with Core Model rank <= this value (default: 3)")
    parser.add_argument("--time-limit", type=int, default=120,
                      help="Maximum time allowed for the solver in seconds (default: 120)")
    parser.add_argument("--increment", type=float, default=0.5,
                      help="Allowed increment for position sizes (default: 0.5)")
    parser.add_argument("--continuous-weights", action="store_true",
                      help="Use continuous weights instead of discrete increments")
    
    args = parser.parse_args()
    
    # Set increment to None if using continuous weights
    increment = None if args.continuous_weights else args.increment
    
    # Run the optimizer
    new_portfolio, added_stocks, optimized_active_share, output_file = main(
        data_file_path=args.data_file,
        num_positions=args.num_positions,
        target_active_share=args.target_active_share,
        sector_tolerance=args.sector_tolerance,
        min_position=args.min_position,
        max_position=args.max_position,
        core_rank_limit=args.core_rank_limit,
        time_limit=args.time_limit,
        increment=increment
    )
    
    if optimized_active_share is None:
        print("\nNo optimal solution found. Check the output file for details.")
    else:
        print(f"\nOptimization complete!")
        print(f"Original Active Share: {args.target_active_share * 100:.2f}%")
        print(f"Optimized Active Share: {optimized_active_share:.2f}%")
        print(f"Results saved to: {output_file}") 