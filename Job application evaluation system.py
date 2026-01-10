import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class JobApplicantEvaluator:
    """
    C5.0 based system for evaluating job applicants
    """
    
    def __init__(self, job_position: str, required_skills: List[str], 
                 experience_weight: float = 0.3, skill_weight: float = 0.4,
                 culture_weight: float = 0.3):
        self.job_position = job_position
        self.required_skills = required_skills
        self.experience_weight = experience_weight
        self.skill_weight = skill_weight
        self.culture_weight = culture_weight
        self.decision_tree = None
        self.feature_importance = {}
        
    def preprocess_applicant_data(self, applicants_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess applicant data for C5.0 algorithm
        """
        processed_data = applicants_df.copy()
        
        # 1. Calculate experience score
        processed_data['experience_score'] = processed_data.apply(
            self._calculate_experience_score, axis=1
        )
        
        # 2. Calculate skill match score
        processed_data['skill_match_score'] = processed_data.apply(
            self._calculate_skill_match_score, axis=1
        )
        
        # 3. Calculate culture fit score
        processed_data['culture_fit_score'] = processed_data.apply(
            self._calculate_culture_fit_score, axis=1
        )
        
        # 4. Calculate overall suitability score
        processed_data['overall_score'] = (
            processed_data['experience_score'] * self.experience_weight +
            processed_data['skill_match_score'] * self.skill_weight +
            processed_data['culture_fit_score'] * self.culture_weight
        )
        
        # 5. Create target variable (Hire/Reject)
        processed_data['hire_recommendation'] = processed_data['overall_score'].apply(
            lambda x: 'Hire' if x >= 0.7 else ('Maybe' if x >= 0.5 else 'Reject')
        )
        
        return processed_data
    
    def _calculate_experience_score(self, applicant: pd.Series) -> float:
        """Calculate normalized experience score (0-1)"""
        try:
            # Years of experience
            years_exp = min(applicant.get('years_experience', 0) / 10, 1.0)
            
            # Relevant experience in similar roles
            relevant_exp = applicant.get('relevant_experience_ratio', 0)
            
            # Company tier/ranking
            company_tier = self._normalize_company_tier(applicant.get('previous_company_tier', 'Unknown'))
            
            # Project complexity
            project_complexity = applicant.get('complex_projects_completed', 0) / 10
            
            return (years_exp * 0.4 + relevant_exp * 0.3 + 
                   company_tier * 0.2 + project_complexity * 0.1)
        except:
            return 0.0
    
    def _calculate_skill_match_score(self, applicant: pd.Series) -> float:
        """Calculate skill match percentage"""
        try:
            applicant_skills = applicant.get('skills', [])
            if isinstance(applicant_skills, str):
                applicant_skills = [s.strip() for s in applicant_skills.split(',')]
            
            # Required skills match
            required_matches = sum(1 for skill in self.required_skills 
                                 if skill.lower() in [s.lower() for s in applicant_skills])
            required_score = required_matches / len(self.required_skills)
            
            # Bonus for additional relevant skills
            additional_skills = applicant.get('additional_skills', [])
            if isinstance(additional_skills, str):
                additional_skills = [s.strip() for s in additional_skills.split(',')]
            
            bonus_skills = len([s for s in additional_skills 
                              if s not in self.required_skills])
            bonus_score = min(bonus_skills * 0.05, 0.2)  # Max 20% bonus
            
            # Certification score
            certifications = applicant.get('certifications', 0)
            cert_score = min(certifications * 0.1, 0.3)  # Max 30% from certs
            
            return min(required_score + bonus_score + cert_score, 1.0)
        except:
            return 0.0
    
    def _calculate_culture_fit_score(self, applicant: pd.Series) -> float:
        """Calculate culture fit score"""
        try:
            scores = []
            
            # Values alignment
            company_values = ['innovation', 'teamwork', 'integrity', 'customer_focus']
            applicant_values = applicant.get('core_values', [])
            if isinstance(applicant_values, str):
                applicant_values = [v.strip() for v in applicant_values.split(',')]
            
            values_match = sum(1 for v in company_values 
                             if v in [av.lower() for av in applicant_values])
            scores.append(values_match / len(company_values) * 0.3)
            
            # Communication style
            comm_style = applicant.get('communication_style', '')
            if comm_style in ['assertive', 'collaborative']:
                scores.append(0.2)
            elif comm_style in ['passive', 'aggressive']:
                scores.append(0.0)
            else:
                scores.append(0.1)
            
            # Leadership potential
            leadership = applicant.get('leadership_potential', 0)
            scores.append(min(leadership * 0.1, 0.2))
            
            # Adaptability
            adaptability = applicant.get('adaptability_score', 0)
            scores.append(min(adaptability, 1.0) * 0.2)
            
            # Reference checks
            ref_score = applicant.get('reference_score', 0)
            scores.append(ref_score * 0.1)
            
            return min(sum(scores), 1.0)
        except:
            return 0.0
    
    def _normalize_company_tier(self, tier: str) -> float:
        """Normalize company tier to 0-1 scale"""
        tier_mapping = {
            'FAANG': 1.0,
            'Fortune 500': 0.9,
            'Tech Unicorn': 0.8,
            'Public Company': 0.7,
            'Large Private': 0.6,
            'Medium Enterprise': 0.5,
            'Small Business': 0.4,
            'Startup': 0.3,
            'Unknown': 0.2
        }
        return tier_mapping.get(tier, 0.2)
    
    def build_decision_tree(self, training_data: pd.DataFrame):
        """
        Build C5.0 decision tree for applicant evaluation
        """
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score, classification_report
        
        # Prepare features
        feature_columns = [
            'years_experience',
            'education_level',
            'skill_match_score',
            'culture_fit_score',
            'expected_salary',
            'notice_period',
            'relocation_willingness'
        ]
        
        # Convert categorical features
        X = pd.get_dummies(training_data[feature_columns], drop_first=True)
        y = training_data['hire_recommendation']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Build C5.0-like decision tree (using scikit-learn for demonstration)
        self.decision_tree = DecisionTreeClassifier(
            criterion='entropy',  # Similar to C5.0
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        
        self.decision_tree.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.decision_tree.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Decision Tree Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        self.feature_importance = dict(zip(X.columns, self.decision_tree.feature_importances_))
        
        return self.decision_tree
    
    def evaluate_applicant(self, applicant_data: Dict) -> Dict:
        """
        Evaluate a single applicant using the decision tree
        """
        if self.decision_tree is None:
            raise ValueError("Decision tree not built. Call build_decision_tree first.")
        
        # Preprocess applicant
        applicant_df = pd.DataFrame([applicant_data])
        processed = self.preprocess_applicant_data(applicant_df)
        
        # Prepare features
        feature_columns = [
            'years_experience',
            'education_level',
            'skill_match_score',
            'culture_fit_score',
            'expected_salary',
            'notice_period',
            'relocation_willingness'
        ]
        
        X = pd.get_dummies(pd.DataFrame([processed.iloc[0][feature_columns]]))
        
        # Ensure all features match training
        # (In production, you'd need to align columns properly)
        
        # Predict
        recommendation = self.decision_tree.predict(X)[0]
        probabilities = self.decision_tree.predict_proba(X)[0]
        
        return {
            'applicant_id': applicant_data.get('applicant_id'),
            'name': applicant_data.get('name'),
            'recommendation': recommendation,
            'confidence': max(probabilities),
            'score_breakdown': {
                'experience_score': float(processed['experience_score'].iloc[0]),
                'skill_match_score': float(processed['skill_match_score'].iloc[0]),
                'culture_fit_score': float(processed['culture_fit_score'].iloc[0]),
                'overall_score': float(processed['overall_score'].iloc[0])
            },
            'key_factors': self._get_key_factors(applicant_data)
        }
    
    def _get_key_factors(self, applicant: Dict) -> List[str]:
        """Identify key positive/negative factors for applicant"""
        factors = []
        
        # Experience factors
        if applicant.get('years_experience', 0) >= 5:
            factors.append("Strong experience level")
        elif applicant.get('years_experience', 0) < 2:
            factors.append("Limited experience")
        
        # Skill factors
        applicant_skills = applicant.get('skills', [])
        if isinstance(applicant_skills, str):
            applicant_skills = [s.strip() for s in applicant_skills.split(',')]
        
        missing_required = [skill for skill in self.required_skills 
                          if skill.lower() not in [s.lower() for s in applicant_skills]]
        if missing_required:
            factors.append(f"Missing key skills: {', '.join(missing_required[:3])}")
        
        # Salary factors
        expected_salary = applicant.get('expected_salary', 0)
        market_average = applicant.get('market_average_salary', 70000)
        if expected_salary > market_average * 1.2:
            factors.append("Salary expectations above market")
        elif expected_salary < market_average * 0.8:
            factors.append("Salary expectations below market (potential red flag)")
        
        return factors
    
    def rank_applicants(self, applicants_data: List[Dict]) -> List[Dict]:
        """
        Rank multiple applicants based on C5.0 evaluation
        """
        evaluations = []
        
        for applicant in applicants_data:
            evaluation = self.evaluate_applicant(applicant)
            evaluations.append(evaluation)
        
        # Sort by overall score
        ranked = sorted(evaluations, 
                       key=lambda x: x['score_breakdown']['overall_score'], 
                       reverse=True)
        
        return ranked
    
    def generate_explanation(self, applicant_evaluation: Dict) -> str:
        """
        Generate human-readable explanation for the decision
        """
        score = applicant_evaluation['score_breakdown']
        factors = applicant_evaluation['key_factors']
        
        explanation = f"""
        Applicant Evaluation for {applicant_evaluation['name']}
        ===================================================
        
        Recommendation: {applicant_evaluation['recommendation']}
        Confidence: {applicant_evaluation['confidence']:.1%}
        
        Score Breakdown:
        • Experience Score: {score['experience_score']:.1%}
        • Skill Match: {score['skill_match_score']:.1%}
        • Culture Fit: {score['culture_fit_score']:.1%}
        • Overall: {score['overall_score']:.1%}
        
        Key Factors:
        """
        
        for factor in factors:
            explanation += f"  • {factor}\n"
        
        if applicant_evaluation['recommendation'] == 'Hire':
            explanation += "\nStrengths: Strong match with position requirements.\n"
        elif applicant_evaluation['recommendation'] == 'Maybe':
            explanation += "\nConsideration: Moderate match - interview recommended.\n"
        else:
            explanation += "\nAreas for Improvement: Does not meet minimum thresholds.\n"
        
        return explanation