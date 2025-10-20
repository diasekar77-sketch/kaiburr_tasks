#!/usr/bin/env python3
"""
Main script to run the complete consumer complaint analysis
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from consumer_complaint_analysis import ConsumerComplaintAnalyzer

def main():
    """Run the complete analysis pipeline"""
    
    print("Consumer Complaint Text Classification Analysis")
    print("=" * 50)
    
    # Check if data exists
    data_file = Path("data/consumer_complaints.csv")
    if not data_file.exists():
        print("Data file not found. Please run download_data.py first.")
        print("Or manually download the data from:")
        print("https://catalog.data.gov/dataset/consumer-complaint-database")
        return
    
    # Initialize analyzer
    analyzer = ConsumerComplaintAnalyzer(str(data_file))
    
    # Run complete analysis
    try:
        analyzer.run_complete_analysis()
        
        # Example prediction
        print("\n" + "=" * 50)
        print("EXAMPLE PREDICTION")
        print("=" * 50)
        
        sample_complaint = """
        I have been trying to get my credit report fixed for months. 
        There are errors on my report that are affecting my ability to get a loan.
        The credit bureau has not responded to my disputes.
        """
        
        print(f"Sample Complaint: {sample_complaint.strip()}")
        
        predictions = analyzer.predict_new_complaint(sample_complaint)
        if predictions:
            print(f"\nPredictions:")
            for model_name, prediction in predictions.items():
                category_names = {
                    0: "Credit reporting, repair, or other",
                    1: "Debt collection", 
                    2: "Consumer Loan",
                    3: "Mortgage"
                }
                print(f"  {model_name}: {category_names.get(prediction, 'Unknown')}")
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

