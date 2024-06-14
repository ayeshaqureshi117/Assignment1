import pytest
from model import load_data, train_model

def test_load_data():
    data = load_data()
    assert data.shape == (150, 5)  # Iris dataset has 150 samples and 5 columns

def test_train_model():
    data = load_data()
    model = train_model(data)
    assert model is not None

if __name__ == "__main__":
    pytest.main()
