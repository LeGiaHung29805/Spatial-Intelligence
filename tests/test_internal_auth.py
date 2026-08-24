import unittest

from src.api_clients.internal_auth import has_valid_internal_token


class InternalAuthTests(unittest.TestCase):
    def test_accepts_only_matching_configured_tokens(self):
        self.assertTrue(has_valid_internal_token("shared-token", "shared-token"))
        self.assertFalse(has_valid_internal_token("shared-token", "wrong-token"))
        self.assertFalse(has_valid_internal_token("", "shared-token"))
        self.assertFalse(has_valid_internal_token("shared-token", None))


if __name__ == "__main__":
    unittest.main()
