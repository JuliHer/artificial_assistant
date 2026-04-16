import json
import importlib
from pathlib import Path
from src.core.skill import Skill

def load_skills(skills_path="src/skills"):
    skills = []

    for skill_dir in Path(skills_path).iterdir():
        if not skill_dir.is_dir():
            continue

        manifest_path = skill_dir / "manifest.json"
        
        if not manifest_path.exists():
            continue

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        skill = _load_single_skill(skill_dir.name, manifest)
        if skill:
            skills.append(skill)

    return sorted(skills, key=lambda s: s.priority, reverse=True)


def _load_single_skill(skill_name, manifest):
    try:
        module_path = f"src.skills.{skill_name}.handler"
        module = importlib.import_module(module_path)

        handler_class = getattr(module, _handler_class_name(skill_name))
        handler_instance = handler_class()

        return Skill(
            name=manifest["name"],
            intents=manifest["intents"],  
            skill_type=manifest["type"],
            handler=handler_instance,
            confirmation=manifest.get("confirmation", False),
            priority=manifest.get("priority", 0)
        )

    except Exception as e:
        print(f"Error cargando skill {skill_name}: {e}")
        return None
    
def _handler_class_name(skill_name):
    return "".join(
        word.capitalize() for word in skill_name.split("_")
    ) + "Handler"