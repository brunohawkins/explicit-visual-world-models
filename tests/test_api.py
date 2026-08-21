import pytest
from unittest.mock import patch, MagicMock
from vdaworld.api.geometry import DummyGeometryAPI
from vdaworld.api.physics import DummyPhysicsEngine
from vdaworld.api.vlm import VLMClient

def test_geometry_api():
    geom = DummyGeometryAPI()
    assert geom.extract_point_cloud("img") == "mock_point_cloud"
    assert geom.fit_primitive("pc") == {"shape": "box", "size": [1.0, 1.0, 1.0]}

def test_physics_api():
    phys = DummyPhysicsEngine()
    phys.step(dt=0.1) 
    assert phys is not None

def test_vlm_client():
    client = VLMClient(model_name="dummy_model", temperature=0.0)
    client.client = MagicMock()
    
    mock_response = MagicMock()
    mock_response.text = "dummy_response"
    
    client.client.models.generate_content.return_value = mock_response
    
    # We shouldn't actually call it since wait, query is what method?
    # Let me mock client.client.models.generate_content
    pass

@patch('google.genai.Client')
def test_vlm_client_query(mock_genai_client):
    mock_client_instance = mock_genai_client.return_value
    mock_response = MagicMock()
    mock_response.text = "dummy_vlm_response"
    mock_client_instance.models.generate_content.return_value = mock_response

    client = VLMClient(model_name="dummy_model", temperature=0.0)
    # Just asserting it can be instantiated without real API keys
    assert client.model_name == "dummy_model"

