import unittest
from autoencoder_lab.utils.config_parser import load_config
from autoencoder_lab.utils.logger import get_logger
from autoencoder_lab.utils.exceptions import ApplicationError

class TestUtils(unittest.TestCase):
    def test_config(self):
        conf = load_config()
        self.assertIn('env', conf)
        
    def test_logger(self):
        logger = get_logger("test")
        self.assertIsNotNone(logger)
        
if __name__ == '__main__':
    unittest.main()
