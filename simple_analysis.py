#!/usr/bin/env python3
"""
Simplified Consumer Complaint Analysis
Works with basic Python libraries (no external dependencies)
"""

import json
import re
from collections import Counter
from datetime import datetime

class SimpleComplaintAnalyzer:
    def __init__(self):
        """Initialize the analyzer"""
        self.complaints = []
        self.categories = {
            0: "Credit reporting, repair, or other",
            1: "Debt collection", 
            2: "Consumer Loan",
            3: "Mortgage"
        }
        
    def create_sample_data(self):
        """Create sample complaint data for demonstration"""
        sample_complaints = [
            {
                "id": 1,
                "product": "Credit reporting, repair, or other",
                "complaint": "I have errors on my credit report that need to be fixed. The credit bureau is not responding to my disputes.",
                "company": "Credit Bureau Inc",
                "state": "CA"
            },
            {
                "id": 2,
                "product": "Debt collection",
                "complaint": "I am being harassed by debt collectors for a debt I do not owe. They call me multiple times a day.",
                "company": "Debt Collectors LLC",
                "state": "NY"
            },
            {
                "id": 3,
                "product": "Consumer Loan",
                "complaint": "I was denied a personal loan due to incorrect information on my credit report. The bank refused to reconsider.",
                "company": "Bank of America",
                "state": "TX"
            },
            {
                "id": 4,
                "product": "Mortgage",
                "complaint": "My mortgage application was rejected due to credit score issues. I believe there are errors in my credit report.",
                "company": "Wells Fargo",
                "state": "FL"
            },
            {
                "id": 5,
                "product": "Credit reporting, repair, or other",
                "complaint": "I found fraudulent accounts on my credit report. I need help removing them immediately.",
                "company": "Equifax",
                "state": "IL"
            }
        ]
        
        self.complaints = sample_complaints
        print(f"✅ Created {len(self.complaints)} sample complaints")
        
    def analyze_data(self):
        """Perform basic data analysis"""
        print("\n" + "="*50)
        print("CONSUMER COMPLAINT ANALYSIS")
        print("="*50)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Analyst: Divya K S")
        print(f"Email: divya.sekar4428@gmail.com")
        print(f"Total Complaints: {len(self.complaints)}")
        
        # Product distribution
        print("\n📊 PRODUCT DISTRIBUTION:")
        product_counts = Counter(complaint['product'] for complaint in self.complaints)
        for product, count in product_counts.items():
            percentage = (count / len(self.complaints)) * 100
            print(f"  {product}: {count} ({percentage:.1f}%)")
        
        # State distribution
        print("\n🌍 STATE DISTRIBUTION:")
        state_counts = Counter(complaint['state'] for complaint in self.complaints)
        for state, count in state_counts.items():
            print(f"  {state}: {count}")
        
        # Company distribution
        print("\n🏢 COMPANY DISTRIBUTION:")
        company_counts = Counter(complaint['company'] for complaint in self.complaints)
        for company, count in company_counts.items():
            print(f"  {company}: {count}")
        
    def text_analysis(self):
        """Perform basic text analysis"""
        print("\n📝 TEXT ANALYSIS:")
        
        # Combine all complaints
        all_text = " ".join(complaint['complaint'] for complaint in self.complaints)
        
        # Word count
        words = re.findall(r'\b\w+\b', all_text.lower())
        word_count = len(words)
        unique_words = len(set(words))
        
        print(f"  Total words: {word_count}")
        print(f"  Unique words: {unique_words}")
        
        # Most common words
        word_counts = Counter(words)
        common_words = word_counts.most_common(10)
        
        print("\n  Most common words:")
        for word, count in common_words:
            if len(word) > 3:  # Filter out short words
                print(f"    {word}: {count}")
        
        # Complaint length analysis
        complaint_lengths = [len(complaint['complaint']) for complaint in self.complaints]
        avg_length = sum(complaint_lengths) / len(complaint_lengths)
        
        print(f"\n  Average complaint length: {avg_length:.1f} characters")
        print(f"  Shortest complaint: {min(complaint_lengths)} characters")
        print(f"  Longest complaint: {max(complaint_lengths)} characters")
        
    def simple_classification(self):
        """Simple keyword-based classification"""
        print("\n🤖 SIMPLE CLASSIFICATION:")
        
        # Define keywords for each category
        keywords = {
            "Credit reporting, repair, or other": ["credit", "report", "bureau", "score", "dispute"],
            "Debt collection": ["debt", "collector", "harass", "call", "payment"],
            "Consumer Loan": ["loan", "denied", "application", "personal", "bank"],
            "Mortgage": ["mortgage", "home", "house", "property", "refinance"]
        }
        
        correct_predictions = 0
        total_predictions = len(self.complaints)
        
        print("  Classification Results:")
        for complaint in self.complaints:
            actual_category = complaint['product']
            complaint_text = complaint['complaint'].lower()
            
            # Find best matching category
            best_match = None
            best_score = 0
            
            for category, category_keywords in keywords.items():
                score = sum(1 for keyword in category_keywords if keyword in complaint_text)
                if score > best_score:
                    best_score = score
                    best_match = category
            
            predicted_category = best_match if best_match else "Unknown"
            is_correct = predicted_category == actual_category
            
            if is_correct:
                correct_predictions += 1
                
            print(f"    Complaint {complaint['id']}: {actual_category} -> {predicted_category} {'✅' if is_correct else '❌'}")
        
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\n  Classification Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
        
    def generate_report(self):
        """Generate analysis report"""
        print("\n📋 ANALYSIS REPORT:")
        print("="*30)
        
        # Summary statistics
        total_complaints = len(self.complaints)
        unique_companies = len(set(complaint['company'] for complaint in self.complaints))
        unique_states = len(set(complaint['state'] for complaint in self.complaints))
        
        print(f"Total Complaints Analyzed: {total_complaints}")
        print(f"Unique Companies: {unique_companies}")
        print(f"Unique States: {unique_states}")
        
        # Most common issues
        all_words = []
        for complaint in self.complaints:
            words = re.findall(r'\b\w+\b', complaint['complaint'].lower())
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        common_issues = [word for word, count in word_counts.most_common(5) if len(word) > 4]
        
        print(f"Most Common Issues: {', '.join(common_issues)}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("1. Implement automated complaint routing based on keywords")
        print("2. Create response templates for common issues")
        print("3. Monitor complaint trends by product category")
        print("4. Improve customer service for high-volume companies")
        
    def run_analysis(self):
        """Run complete analysis"""
        print("Consumer Complaint Analysis - Simplified Version")
        print("="*60)
        print(f"Analyst: Divya K S")
        print(f"Email: divya.sekar4428@gmail.com")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Create sample data
        self.create_sample_data()
        
        # Perform analysis
        self.analyze_data()
        self.text_analysis()
        self.simple_classification()
        self.generate_report()
        
        print("\n✅ Analysis completed successfully!")
        print("Note: This is a simplified version for demonstration purposes.")
        print("For production use, install required packages: pip install -r requirements.txt")

def main():
    """Main function"""
    analyzer = SimpleComplaintAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
