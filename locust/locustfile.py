
from locust import HttpUser, task, between
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
    PROJECT_ROOT /
    "data" /
    "load_test_images" /
    "sample.jpg"
)


class GeoVisionUser(HttpUser):

    # Each simulated user waits 1–3 seconds
    # between requests.
    wait_time = between(1, 3)

    def on_start(self):

        if not IMAGE_PATH.exists():
            raise FileNotFoundError(
                f"Load test image not found: {IMAGE_PATH}"
            )

        self.image_bytes = IMAGE_PATH.read_bytes()

    @task
    def predict_satellite_image(self):

        self.client.post(
            "/predict",
            files={
                "file": (
                    "sample.jpg",
                    self.image_bytes,
                    "image/jpeg"
                )
            },
            name="/predict"
        )

    @task(2)
    def check_api_health(self):

        self.client.get(
            "/health",
            name="/health"
        )
