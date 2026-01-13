#!/bin/bash

# Parameter testing script for pulmonary tree graph labeling
# Tests all combinations of diameter and length thresholds

# Base command parameters
BASE_CMD="python scripts/batch_process_patients.py"
DATA_DIR="data"
PATIENT_LIST="./data/sample_24_patients_list.txt"
BASE_OUTPUT_DIR="/home/desligneris/Documents/0_test_for_manuscript/pipeline/test_on_24_patients/test_parameters_labelling"
COORDINATE_FORMAT="both"
SKIP_STEP="embolism"
LOG_LEVEL="DEBUG"

# Parameter arrays to test
DIAMETER_THRESHOLDS=(0.5 0.6 0.7)
LENGTH_THRESHOLDS=(1.5 2.0 2.5)

# Colors for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log with timestamp
log_with_timestamp() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Function to create output directory name
get_output_dir() {
    local diameter=$1
    local length=$2
    # Replace dots with underscores for directory names
    local diameter_str=$(echo $diameter | sed 's/\./_/g')
    local length_str=$(echo $length | sed 's/\./_/g')
    echo "${BASE_OUTPUT_DIR}/test_d${diameter_str}_l${length_str}"
}

# Function to run single test
run_test() {
    local diameter=$1
    local length=$2
    local output_dir=$(get_output_dir $diameter $length)
    local test_name="d${diameter}_l${length}"

    log_with_timestamp "${YELLOW}Starting test: $test_name${NC}"
    log_with_timestamp "Output directory: $output_dir"

    # Create output directory
    mkdir -p "$output_dir"

    # Build command
    local cmd="$BASE_CMD \
        --data-dir $DATA_DIR \
        --patient-list $PATIENT_LIST \
        --coordinate-format $COORDINATE_FORMAT \
        --save-skeletons \
        --output-dir $output_dir \
        --skip-step $SKIP_STEP \
        --log-level $LOG_LEVEL \
        --diameter-threshold $diameter \
        --length-threshold $length"

    # Log the command
    log_with_timestamp "Command: $cmd"

    # Record start time
    local start_time=$(date +%s)

    # Run the command and capture exit status
    if eval $cmd; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_with_timestamp "${GREEN}✓ Test $test_name completed successfully in ${duration}s${NC}"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_with_timestamp "${RED}✗ Test $test_name failed after ${duration}s${NC}"
        return 1
    fi
}

# Main execution
main() {
    log_with_timestamp "${BLUE}Starting parameter testing script${NC}"
    log_with_timestamp "Testing ${#DIAMETER_THRESHOLDS[@]} diameter thresholds × ${#LENGTH_THRESHOLDS[@]} length thresholds = $((${#DIAMETER_THRESHOLDS[@]} * ${#LENGTH_THRESHOLDS[@]})) combinations"

    local total_tests=0
    local successful_tests=0
    local failed_tests=0
    local overall_start_time=$(date +%s)

    # Create summary log file
    local summary_file="${BASE_OUTPUT_DIR}/test_summary_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$BASE_OUTPUT_DIR"

    echo "Parameter Testing Summary - $(date)" > "$summary_file"
    echo "=================================" >> "$summary_file"
    echo "" >> "$summary_file"

    # Loop through all combinations
    for diameter in "${DIAMETER_THRESHOLDS[@]}"; do
        for length in "${LENGTH_THRESHOLDS[@]}"; do
            total_tests=$((total_tests + 1))

            log_with_timestamp "${BLUE}=== Test $total_tests/$((${#DIAMETER_THRESHOLDS[@]} * ${#LENGTH_THRESHOLDS[@]})) ===${NC}"

            if run_test $diameter $length; then
                successful_tests=$((successful_tests + 1))
                echo "✓ d${diameter}_l${length}: SUCCESS" >> "$summary_file"
            else
                failed_tests=$((failed_tests + 1))
                echo "✗ d${diameter}_l${length}: FAILED" >> "$summary_file"
            fi

            # Add spacing between tests
            echo ""
        done
    done

    # Calculate overall duration
    local overall_end_time=$(date +%s)
    local overall_duration=$((overall_end_time - overall_start_time))
    local avg_duration=$((overall_duration / total_tests))

    # Print final summary
    log_with_timestamp "${BLUE}=== FINAL SUMMARY ===${NC}"
    log_with_timestamp "Total tests run: $total_tests"
    log_with_timestamp "${GREEN}Successful: $successful_tests${NC}"
    log_with_timestamp "${RED}Failed: $failed_tests${NC}"
    log_with_timestamp "Total time: ${overall_duration}s ($(($overall_duration / 3600))h $(($overall_duration % 3600 / 60))m)"
    log_with_timestamp "Average time per test: ${avg_duration}s"
    log_with_timestamp "Summary saved to: $summary_file"

    # Add summary to log file
    echo "" >> "$summary_file"
    echo "Final Results:" >> "$summary_file"
    echo "Total tests: $total_tests" >> "$summary_file"
    echo "Successful: $successful_tests" >> "$summary_file"
    echo "Failed: $failed_tests" >> "$summary_file"
    echo "Total time: ${overall_duration}s" >> "$summary_file"
    echo "Average time per test: ${avg_duration}s" >> "$summary_file"

    # Show directory structure
    log_with_timestamp "${YELLOW}Output directory structure:${NC}"
    find "$BASE_OUTPUT_DIR" -maxdepth 2 -type d | sort

    if [ $failed_tests -eq 0 ]; then
        log_with_timestamp "${GREEN}All tests completed successfully!${NC}"
        exit 0
    else
        log_with_timestamp "${RED}Some tests failed. Check the summary for details.${NC}"
        exit 1
    fi
}

# Handle script interruption
cleanup() {
    log_with_timestamp "${RED}Script interrupted. Cleaning up...${NC}"
    exit 130
}

trap cleanup SIGINT SIGTERM

# Run main function
main "$@"
