class ApplicantDataGenerator:
    """Generate sample applicant data for testing"""
    
    @staticmethod
    def generate_applicants(num_applicants=50):
        """Generate synthetic applicant data"""
        import random
        from faker import Faker
        
        fake = Faker()
        applicants = []
        
        positions = ['Software Engineer', 'Data Scientist', 'Product Manager', 
                    'UX Designer', 'DevOps Engineer']
        skills_sets = {
            'Software Engineer': ['Python', 'Java', 'JavaScript', 'AWS', 'Docker', 'SQL'],
            'Data Scientist': ['Python', 'Machine Learning', 'SQL', 'Statistics', 'TensorFlow'],
            'Product Manager': ['Agile', 'Scrum', 'Product Strategy', 'User Research', 'SQL'],
            'UX Designer': ['Figma', 'User Research', 'Wireframing', 'Prototyping', 'UI Design'],
            'DevOps Engineer': ['AWS', 'Docker', 'Kubernetes', 'CI/CD', 'Linux', 'Python']
        }
        
        for i in range(num_applicants):
            position = random.choice(positions)
            skills = skills_sets[position]
            
            applicant = {
                'applicant_id': f"APP{1000 + i}",
                'name': fake.name(),
                'email': fake.email(),
                'phone': fake.phone_number(),
                'position_applied': position,
                'years_experience': random.randint(1, 15),
                'education_level': random.choice(['Bachelors', 'Masters', 'PhD', 'Bootcamp']),
                'skills': ', '.join(random.sample(skills, random.randint(3, len(skills)))),
                'additional_skills': ', '.join(random.sample(['Git', 'Agile', 'Scrum', 'AWS', 'Docker'], 
                                                           random.randint(0, 3))),
                'certifications': random.randint(0, 5),
                'previous_company_tier': random.choice(['FAANG', 'Fortune 500', 'Tech Unicorn', 
                                                       'Public Company', 'Startup']),
                'relevant_experience_ratio': random.uniform(0.3, 1.0),
                'complex_projects_completed': random.randint(0, 10),
                'core_values': ', '.join(random.sample(['innovation', 'teamwork', 'integrity', 
                                                       'customer_focus', 'excellence'], 2)),
                'communication_style': random.choice(['assertive', 'collaborative', 'passive', 'aggressive']),
                'leadership_potential': random.randint(1, 5),
                'adaptability_score': random.uniform(0.0, 1.0),
                'reference_score': random.uniform(0.5, 1.0),
                'expected_salary': random.randint(60000, 150000),
                'notice_period': random.randint(14, 90),  # days
                'relocation_willingness': random.choice([True, False]),
                'application_date': fake.date_this_year()
            }
            applicants.append(applicant)
        
        return applicants