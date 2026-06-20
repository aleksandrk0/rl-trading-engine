import unittest
import torch
import sys
from pathlib import Path
import logging

# Добавляем путь к проекту
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backtest import ModelLoader
from models.feature_net import FeatureExtractor
from models.directional_net import DirectionalPredictor
from models.regime_net import RegimeDetector
from models.qnd_net import QNDAgent
from models.ensemble_net import EnhancedEnsemble

class TestModelLoader(unittest.TestCase):
    def setUp(self):
        """Инициализация тестов"""
        self.models_dir = project_root / "models"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Настройка логирования
        self.logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Инициализация загрузчика
        self.loader = ModelLoader(str(self.models_dir))
        
        # Ожидаемые размерности
        self.output_dims = {
            'feature': (1, 60, 2048),
            'directional': (1, 2),
            'regime': (1, 5),
            'qnd': (1, 3),
            'ensemble': (1, 2)
        }

    def test_model_dimensions(self):
        """Тест размерностей моделей"""
        for name in self.loader.model_sequence:
            try:
                self.logger.info(f"\nTesting {name} model dimensions...")
                
                # Загрузка модели
                model = self.loader._load_model(name)
                
                # Проверка выходов
                with torch.no_grad():
                    if name == 'feature':
                        x = torch.randn(1, 60, 133, device=self.device)
                        out = model(x)
                        self.assertEqual(
                            out['features'].shape,
                            self.output_dims[name]
                        )
                    else:
                        if name == 'ensemble':
                            inputs = {
                                'directional': torch.randn(1, 2, device=self.device),
                                'regime': torch.randn(1, 5, device=self.device),
                                'qnd': torch.randn(1, 3, device=self.device)
                            }
                        else:
                            inputs = torch.randn(
                                1, 
                                60, 
                                self.loader.model_dims[name]['input'],
                                device=self.device
                            )
                        out = model(inputs)
                        self.assertEqual(
                            out['logits'].shape,
                            self.output_dims[name]
                        )

            except Exception as e:
                self.fail(f"Error testing {name}: {str(e)}")

    def test_model_loading(self):
        """Тест загрузки всех моделей"""
        try:
            self.loader.load_models()
            
            # Тест обработки
            x = torch.randn(1, 60, 133, device=self.device)
            outputs = self.loader.process_features(x)
            
            # Проверка выходов
            for name, dims in self.output_dims.items():
                self.assertIn(name, outputs)
                out = outputs[name]
                if isinstance(out, dict):
                    main_out = out['features'] if 'features' in out else out['logits']
                else:
                    main_out = out
                self.assertEqual(
                    tuple(main_out.shape),
                    dims,
                    f"Wrong output dims for {name}"
                )
                
        except Exception as e:
            self.fail(f"Error in model loading: {str(e)}")

    def tearDown(self):
        """Очистка ресурсов"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    unittest.main()