import pandas as pd
import pytest
from neer_match_utilities.panel import GenerateID  # Correct import path for GenerateID class

# Mock class for testing relations_left_right
class MockGenerateID(GenerateID):
    def __init__(self):
        self.relation = 'm:m'

# Test relations_left_right for all validation modes
def test_relations_left_right():
    # Mock GenerateID instance
    gid = MockGenerateID()

    # Input DataFrame
    df = pd.DataFrame({
        'left': [1, 1, 2, 3],
        'right': [3, 4, 3, 4],
        'prediction': [0.95, 0.9, 0.85, 0.8]
    })

    # no string passed (no duplicates removed)
    result = gid.relations_left_right(df)
    assert len(result) == 4, "m:m should return all predictions without duplicates."

    # 'm:m' (no duplicates removed)
    result = gid.relations_left_right(df, relation='m:m')
    assert len(result) == 4, "m:m should return all predictions without duplicates."

    # '1:m' (unique 'left' values)
    result = gid.relations_left_right(df, relation='1:m')
    assert result['left'].nunique() == len(result), "1:m should ensure unique 'left' values."

    # 'm:1' (unique 'right' values)
    result = gid.relations_left_right(df, relation='m:1')
    assert result['right'].nunique() == len(result), "m:1 should ensure unique 'right' values."

    # '1:1' (unique 'left' and 'right' values)
    result = gid.relations_left_right(df, relation='1:1')
    assert result['left'].nunique() == len(result), "1:1 should ensure unique 'left' values."
    assert result['right'].nunique() == len(result), "1:1 should ensure unique 'right' values."

    # Invalid relation value
    with pytest.raises(ValueError, match=r"Invalid value for `relation`:.*"):
        gid.relations_left_right(df, relation='invalid_option')

if __name__ == "__main__":
    pytest.main()