
from dataclasses import dataclass
from typing import Dict

@dataclass
class Student:
    name: str
    age: int
    grades: Dict[str, float]  # Attribute as a dictionary

    def add_grade(self, subject: str, grade: float):
        """Add or update the grade for a subject."""
        self.grades[subject] = grade

    def average_grade(self) -> float:
        """Calculate the average grade."""
        if not self.grades:
            return 0.0  # Return 0.0 if there are no grades
        
        return sum(self.grades.values()) / len(self.grades)

# Creating an instance of Student with an empty grades dictionary
student = Student(name='Alice', age=20, grades={})

# Adding grades
student.add_grade('Math', 90.0)
student.add_grade('English', 85.5)
student.add_grade('Science', 92.0)

# Accessing the grades and calculating the average
print(f"Grades: {student.grades}")                                 # Output: Grades: {'Math': 90.0, 'English': 85.5, 'Science': 92.0}
print(f"Average Grade: {student.average_grade():.2f}") 