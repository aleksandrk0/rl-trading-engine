import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Добавляем корневую директорию проекта в PYTHONPATH
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

# Теперь импортируем модели
from models.feature_net import FeatureExtractor
from models.directional_net import DirectionalPredictor
from models.regime_net import RegimeDetector
from models.qnd_net import QNDAgent
from models.ensemble_net import EnhancedEnsemble

# Добавляем путь к корневой директории проекта
sys.path.append(str(Path(__file__).parent.parent))

def test_model_dimensions():
    # Устанавливаем параметры для тестового входа
    batch_size = 1
    sequence_length = 300
    input_dim = 133
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nStarting model dimension test on {device}")
    print("-" * 50)

    try:
        # 1. Создаем тестовый вход
        test_input = torch.randn(batch_size, sequence_length, input_dim).to(device)
        print(f"Input tensor shape: {test_input.shape}")

        # 2. FeatureExtractor
        feature_net = FeatureExtractor().to(device)
        with torch.no_grad():
            feature_outputs = feature_net(test_input)
        
        print("\nFeatureExtractor outputs:")
        for key, tensor in feature_outputs.items():
            if isinstance(tensor, torch.Tensor):
                print(f"- {key}: {tensor.shape}")

        # 3. DirectionalPredictor
        directional_net = DirectionalPredictor().to(device)
        with torch.no_grad():
            directional_outputs = directional_net(feature_outputs['features'])
        
        print("\nDirectionalPredictor outputs:")
        for key, tensor in directional_outputs.items():
            if isinstance(tensor, torch.Tensor):
                print(f"- {key}: {tensor.shape}")

        # 4. RegimeDetector
        regime_net = RegimeDetector().to(device)
        with torch.no_grad():
            regime_outputs = regime_net(feature_outputs['features'])
            
        print("\nRegimeDetector outputs:")
        for key, tensor in regime_outputs.items():
            if isinstance(tensor, torch.Tensor):
                print(f"- {key}: {tensor.shape}")

        # 5. QNDAgent
        qnd_net = QNDAgent().to(device)
        with torch.no_grad():
            qnd_outputs = qnd_net(feature_outputs['features'])
            
        print("\nQNDAgent outputs:")
        for key, tensor in qnd_outputs.items():
            if isinstance(tensor, torch.Tensor):
                print(f"- {key}: {tensor.shape}")

        # 6. EnhancedEnsemble
        ensemble_net = EnhancedEnsemble().to(device)
        
        # Подготавливаем входы для ансамбля
        ensemble_inputs = {
            'features': feature_outputs['features'],
            'directional': directional_outputs['logits'],
            'regime': regime_outputs['logits'],
            'qnd': qnd_outputs['logits']
        }
        
        with torch.no_grad():
            ensemble_outputs = ensemble_net(ensemble_inputs)
            
        print("\nEnhancedEnsemble outputs:")
        for key, tensor in ensemble_outputs.items():
            if isinstance(tensor, torch.Tensor):
                print(f"- {key}: {tensor.shape}")

        print("\nAll models dimensions tested successfully!")
        
    except Exception as e:
        print(f"\nError during dimension test: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_model_dimensions()