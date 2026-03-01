"""Example test module"""

import unittest


class TestExample(unittest.TestCase):
    """Example test class"""

    def test_basic(self):
        """Test basic functionality"""
        self.assertEqual(1 + 1, 2)


if __name__ == "__main__":
    unittest.main()
