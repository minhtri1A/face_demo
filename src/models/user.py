from pydantic import BaseModel
from fastapi import Form
from typing import Optional

class User(BaseModel):
    id: str
    name: str
    email: Optional[str] = None

    @classmethod
    def as_form(
        cls,
        id: str = Form(...),
        name: str = Form(...),
        email: Optional[str] = Form(None),
    ):
        return cls(id=id, name=name, email=email)