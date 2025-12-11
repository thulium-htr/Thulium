from typing import Dict

def generate_report(metrics: Dict[str, float], output_format="markdown") -> str:
    """
    Generate a formatted report string.
    """
    lines = ["# Evaluation Report", ""]
    for k, v in metrics.items():
        lines.append(f"- **{k}**: {v:.4f}")
    return "\n".join(lines)
