from app.schemas.profile import ProfileUpdateIn
from app.core.enums import Gender, ExperienceLevel, RunningGoal
from pydantic import ValidationError
import sys

def test_schema():
    print("Testing ProfileUpdateIn Schema...")
    
    # Test Valid Data
    try:
        valid_data = {
            "age": 25,
            "weight": 70.5,
            "gender": "male",
            "experience_level": "beginner",
            "running_goal": "weight_loss",
            "has_injuries": True
        }
        model = ProfileUpdateIn(**valid_data)
        print("✅ VALID data accepted.")
        
        # Test serialization
        dumped = model.model_dump(mode='json')
        print(f"Dumped: {dumped}")
        assert dumped['gender'] == 'male'
        assert dumped['has_injuries'] is True
        print("✅ Serialization correct.")
        
    except Exception as e:
        print(f"❌ VALID data failed: {e}")
        sys.exit(1)

    # Test Invalid Enum
    try:
        invalid_data = {
            "gender": "invalid_gender"
        }
        ProfileUpdateIn(**invalid_data)
        print("❌ INVALID gender accepted (FAIL).")
        sys.exit(1)
    except ValidationError as e:
        print("✅ INVALID gender rejected.")

    # Test Invalid Bool passed as string that isn't a bool representation
    # Pydantic is lenient with bools, "yes" -> True, but "invalid" -> error
    try:
        invalid_bool = {
            "has_injuries": "not_a_bool"
        }
        ProfileUpdateIn(**invalid_bool)
        print("❌ INVALID bool accepted (FAIL).")
        sys.exit(1)
    except ValidationError:
        print("✅ INVALID bool rejected.")

if __name__ == "__main__":
    test_schema()
