class HRDecisionSupportSystem:
    """
    Complete HR Decision Support System using C5.0 algorithm
    """
    
    def __init__(self, company_name: str):
        self.company_name = company_name
        self.position_evaluators = {}
        self.historical_data = pd.DataFrame()
        
    def add_position(self, position: str, requirements: Dict):
        """Add a new position to evaluate"""
        required_skills = requirements.get('required_skills', [])
        evaluator = JobApplicantEvaluator(
            job_position=position,
            required_skills=required_skills,
            experience_weight=requirements.get('experience_weight', 0.3),
            skill_weight=requirements.get('skill_weight', 0.4),
            culture_weight=requirements.get('culture_weight', 0.3)
        )
        self.position_evaluators[position] = evaluator
        return evaluator
    
    def train_with_historical_data(self, historical_data: pd.DataFrame):
        """Train models using historical hiring data"""
        for position in historical_data['position'].unique():
            if position in self.position_evaluators:
                position_data = historical_data[historical_data['position'] == position]
                
                # Add hire recommendation based on actual outcomes
                position_data = position_data.copy()
                position_data['hire_recommendation'] = position_data['hired'].apply(
                    lambda x: 'Hire' if x else ('Maybe' if x is None else 'Reject')
                )
                
                # Train decision tree
                self.position_evaluators[position].build_decision_tree(position_data)
    
    def screen_applicants(self, position: str, applicants: List[Dict]) -> pd.DataFrame:
        """Screen applicants for a specific position"""
        if position not in self.position_evaluators:
            raise ValueError(f"No evaluator found for position: {position}")
        
        evaluator = self.position_evaluators[position]
        
        # Evaluate each applicant
        results = []
        for applicant in applicants:
            if applicant['position_applied'] == position:
                evaluation = evaluator.evaluate_applicant(applicant)
                results.append({
                    'applicant_id': evaluation['applicant_id'],
                    'name': evaluation['name'],
                    'recommendation': evaluation['recommendation'],
                    'confidence': evaluation['confidence'],
                    'overall_score': evaluation['score_breakdown']['overall_score'],
                    'experience_score': evaluation['score_breakdown']['experience_score'],
                    'skill_score': evaluation['score_breakdown']['skill_match_score'],
                    'culture_score': evaluation['score_breakdown']['culture_fit_score'],
                    'key_factors': '; '.join(evaluation['key_factors'])
                })
        
        return pd.DataFrame(results)
    
    def generate_hiring_report(self, position: str, applicants: List[Dict]) -> str:
        """Generate comprehensive hiring report"""
        evaluator = self.position_evaluators[position]
        
        # Rank applicants
        ranked_applicants = evaluator.rank_applicants(
            [a for a in applicants if a['position_applied'] == position]
        )
        
        report = f"""
        HIRING REPORT: {position}
        Company: {self.company_name}
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        ============================================================
        
        Total Applicants: {len(ranked_applicants)}
        """
        
        # Top candidates
        report += "\nTOP CANDIDATES:\n"
        report += "=" * 80 + "\n"
        
        for i, applicant in enumerate(ranked_applicants[:5], 1):
            report += f"\n{i}. {applicant['name']} (ID: {applicant['applicant_id']})\n"
            report += f"   Recommendation: {applicant['recommendation']}\n"
            report += f"   Overall Score: {applicant['score_breakdown']['overall_score']:.1%}\n"
            report += f"   Key Factors: {applicant['key_factors'][0] if applicant['key_factors'] else 'N/A'}\n"
        
        # Statistics
        recommendations = [a['recommendation'] for a in ranked_applicants]
        hire_count = recommendations.count('Hire')
        maybe_count = recommendations.count('Maybe')
        reject_count = recommendations.count('Reject')
        
        report += f"\n\nSUMMARY STATISTICS:\n"
        report += "=" * 80 + "\n"
        report += f"Strong Candidates (Hire): {hire_count}\n"
        report += f"Borderline Candidates (Maybe): {maybe_count}\n"
        report += f"Weak Candidates (Reject): {reject_count}\n"
        
        if len(ranked_applicants) > 0:
            avg_score = np.mean([a['score_breakdown']['overall_score'] 
                               for a in ranked_applicants])
            report += f"Average Applicant Score: {avg_score:.1%}\n"
        
        # Recommendations
        report += f"\n\nRECOMMENDATIONS:\n"
        report += "=" * 80 + "\n"
        
        if hire_count >= 3:
            report += "• Strong candidate pool - proceed with interviews for top 3 candidates\n"
        elif hire_count > 0:
            report += "• Limited strong candidates - interview top candidates, consider reopening applications\n"
        else:
            report += "• No strong candidates found - recommend reopening applications\n"
        
        # Feature importance
        if hasattr(evaluator, 'feature_importance') and evaluator.feature_importance:
            report += f"\nDECISION FACTORS (Feature Importance):\n"
            report += "=" * 80 + "\n"
            
            sorted_features = sorted(evaluator.feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
            
            for feature, importance in sorted_features:
                report += f"• {feature}: {importance:.1%}\n"
        
        return report
    
    def bias_detection(self, applicants_df: pd.DataFrame) -> Dict:
        """
        Detect potential biases in applicant evaluation
        """
        results = {}
        
        # Check for gender bias
        if 'gender' in applicants_df.columns:
            gender_groups = applicants_df.groupby('gender')
            gender_scores = gender_groups['overall_score'].mean()
            results['gender_bias'] = {
                'scores': gender_scores.to_dict(),
                'disparity': gender_scores.max() - gender_scores.min()
            }
        
        # Check for education bias
        if 'education_level' in applicants_df.columns:
            edu_groups = applicants_df.groupby('education_level')
            edu_scores = edu_groups['overall_score'].mean()
            results['education_bias'] = {
                'scores': edu_scores.to_dict(),
                'disparity': edu_scores.max() - edu_scores.min()
            }
        
        return results