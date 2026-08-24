import unittest
from unittest.mock import Mock, patch

from src.api_clients.module2_api import heatmap_client


class HeatmapClientTests(unittest.TestCase):
    def test_does_not_send_request_without_internal_token(self):
        with patch.object(heatmap_client, "INTERNAL_API_TOKEN", ""), patch.object(
            heatmap_client.requests, "post"
        ) as post:
            self.assertFalse(heatmap_client.trigger_heatmap_update())
            post.assert_not_called()

    def test_sends_internal_token_header(self):
        response = Mock(status_code=200)
        with patch.object(heatmap_client, "INTERNAL_API_TOKEN", "test-token"), patch.object(
            heatmap_client.requests, "post", return_value=response
        ) as post:
            self.assertTrue(heatmap_client.trigger_heatmap_update())

        post.assert_called_once_with(
            "http://localhost:8080/api/v1/map/internal/trigger-broadcast",
            headers={"X-Internal-Api-Token": "test-token"},
            timeout=5,
        )


if __name__ == "__main__":
    unittest.main()
