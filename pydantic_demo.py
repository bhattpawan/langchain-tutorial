from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):
    name: str = "Default Name"
    age: int
    email: Optional[EmailStr] = None
    cgpa: float = Field(
        gt=0.0,
        le=10.0,
        default=1.0,
        description="CGPA of the student which must be between 0.0 and 10.0",
    )


student_dict = {"name": "Pawan", "age": 20}
student = Student(**student_dict)

print(student)
