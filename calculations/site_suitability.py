
def calculate_suitability(
    illumination_score=None,
    terrain_score=None,
    earth_visibility_score=None,
    exploration_score=None,
):
    """
    Calculate a preliminary lunar landing site suitability score.

    Each score must be between 0 and 100.
    Missing data is excluded and remaining weights are normalized.
    """

    weights = {
        "Solar illumination": 0.30,
        "Terrain safety": 0.30,
        "Earth visibility": 0.20,
        "Exploration potential": 0.20,
    }

    scores = {
        "Solar illumination": illumination_score,
        "Terrain safety": terrain_score,
        "Earth visibility": earth_visibility_score,
        "Exploration potential": exploration_score,
    }

    available = {
        name: score
        for name, score in scores.items()
        if score is not None
    }

    if not available:
        return {
            "overall_score": None,
            "category_scores": scores,
            "data_completeness": 0,
            "assessment": "Insufficient data",
        }

    for name, score in available.items():
        if not isinstance(score, (int, float)) or not 0 <= score <= 100:
            raise ValueError(f"{name} must be between 0 and 100.")

    total_weight = sum(weights[name] for name in available)

    overall = sum(
        score * weights[name]
        for name, score in available.items()
    ) / total_weight

    completeness = len(available) / len(weights) * 100

    if completeness < 100:
        assessment = "Preliminary — incomplete data"
    elif overall >= 80:
        assessment = "Highly promising"
    elif overall >= 60:
        assessment = "Promising"
    elif overall >= 40:
        assessment = "Moderate"
    else:
        assessment = "Challenging"

    return {
        "overall_score": round(overall, 1),
        "category_scores": scores,
        "data_completeness": round(completeness, 1),
        "assessment": assessment,
    }
