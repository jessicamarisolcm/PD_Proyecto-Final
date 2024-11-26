from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_identify_plant():
    response = client.post(
        "/identify/",
        files={"image": ("test.jpg", open("tests/test_image.jpg", "rb"), "image/jpeg")},
    )
    assert response.status_code == 200
