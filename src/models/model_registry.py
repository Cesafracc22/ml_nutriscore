from typing import Dict, Type, Optional, Any
from pathlib import Path
import json


"""
This class is a registry for the models.
It allows to register new models and to get the models by name, category, etc.
"""
class ModelRegistry:

    _models: Dict[str, Type] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        model_class: Type,
        description: str = "",
        category: str = "general",
        requires_probability: bool = False
    ) -> Type:
        if name in cls._models:
            raise ValueError(f"Model '{name}' is already registered.")

        cls._models[name] = model_class
        cls._metadata[name] = {
            'description': description,
            'category': category,
            'requires_probability': requires_probability,
            'class_name': model_class.__name__
        }

        return model_class

    @classmethod
    def get_model_class(cls, name: str) -> Type:
        if name not in cls._models:
            available = ', '.join(cls._models.keys())
            raise KeyError(
                f"Model '{name}' not found in registry. "
                f"Available models: {available}"
            )

        return cls._models[name]

    @classmethod
    def create_model(cls, name: str, **hyperparameters) -> Any:
        model_class = cls.get_model_class(name)
        return model_class(**hyperparameters)

    @classmethod
    def list_models(cls) -> Dict[str, Dict[str, Any]]:
        return cls._metadata.copy()

    @classmethod
    def get_models_by_category(cls, category: str) -> Dict[str, Type]:
        return {
            name: cls._models[name]
            for name, metadata in cls._metadata.items()
            if metadata['category'] == category
        }

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._models

    @classmethod
    def get_model_info(cls, name: str) -> Optional[Dict[str, Any]]:
        return cls._metadata.get(name)

    @classmethod
    def print_registry(cls) -> None:
        if not cls._models:
            print("No models registered yet.")
            return

        print("\n" + "="*70)
        print("REGISTERED MODELS")
        print("="*70)

        categories = {}
        for name, metadata in cls._metadata.items():
            category = metadata['category']
            if category not in categories:
                categories[category] = []
            categories[category].append((name, metadata))

        for category, models in sorted(categories.items()):
            print(f"\n{category.upper()} MODELS:")
            print("-"*70)
            for name, metadata in sorted(models):
                prob_req = "✓" if metadata['requires_probability'] else "✗"
                print(f"  • {name:25} | Prob: {prob_req} | {metadata['description']}")

        print("\n" + "="*70 + "\n")

    @classmethod
    def save_registry_info(cls, filepath: Path) -> None:
        registry_data = {
            'models': cls._metadata,
            'total_models': len(cls._models)
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(registry_data, f, indent=2)

        print(f"Registry information saved to {filepath}")


def register_model(
    name: str,
    description: str = "",
    category: str = "general",
    requires_probability: bool = False
):
    def decorator(model_class: Type) -> Type:
        ModelRegistry.register(
            name=name,
            model_class=model_class,
            description=description,
            category=category,
            requires_probability=requires_probability
        )
        return model_class

    return decorator
