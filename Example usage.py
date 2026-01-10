# Example usage
if __name__ == "__main__":
    # Initialize HR system
    hr_system = HRDecisionSupportSystem(company_name="TechCorp Inc.")
    
    # Define position requirements
    software_engineer_req = {
        'required_skills': ['Python', 'JavaScript', 'AWS', 'Docker', 'SQL'],
        'experience_weight': 0.3,
        'skill_weight': 0.4,
        'culture_weight': 0.3
    }
    
    # Add position
    evaluator = hr_system.add_position("Software Engineer", software_engineer_req)
    
    # Generate sample applicants
    generator = ApplicantDataGenerator()
    applicants = generator.generate_applicants(20)
    
    # Create historical training data (simulated)
    historical_data = pd.DataFrame(applicants[:10])
    historical_data['hired'] = [True, False, True, False, True, False, True, False, True, False]
    
    # Train the model
    hr_system.train_with_historical_data(historical_data)
    
    # Screen new applicants
    new_applicants = applicants[10:]
    screening_results = hr_system.screen_applicants("Software Engineer", new_applicants)
    
    print("Screening Results:")
    print("=" * 80)
    print(screening_results[['name', 'recommendation', 'overall_score', 'key_factors']])
    
    # Generate hiring report
    report = hr_system.generate_hiring_report("Software Engineer", new_applicants)
    print("\n" + "=" * 80)
    print(report)
    
    # Evaluate specific applicant
    sample_applicant = new_applicants[0]
    evaluation = evaluator.evaluate_applicant(sample_applicant)
    explanation = evaluator.generate_explanation(evaluation)
    
    print("\n" + "=" * 80)
    print("INDIVIDUAL APPLICANT EVALUATION:")
    print(explanation)
    
    # Check for biases
    all_applicants_df = pd.DataFrame(applicants)
    if 'overall_score' in all_applicants_df.columns:
        bias_results = hr_system.bias_detection(all_applicants_df)
        print("\n" + "=" * 80)
        print("BIAS DETECTION RESULTS:")
        print(f"Education bias disparity: {bias_results.get('education_bias', {}).get('disparity', 0):.3f}")