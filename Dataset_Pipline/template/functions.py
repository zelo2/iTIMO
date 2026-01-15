tools = [
        {
            "type": "function",
            "function": {
                "name": "geo_distance_segments",
                "description": "Compute consecutive great-circle distances and Low/Medium/High classes for BEFORE and AFTER itineraries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "waypoints_before": {
                            "type": "array",
                            "items": {"type": "object",
                                      "properties": {"lat": {"type": "number"}, "lon": {"type": "number"}},
                                      "required": ["lat", "lon"]}
                        },
                        "waypoints_after": {
                            "type": "array",
                            "items": {"type": "object",
                                      "properties": {"lat": {"type": "number"}, "lon": {"type": "number"}},
                                      "required": ["lat", "lon"]}
                        }
                    },
                    "required": ["waypoints_before", "waypoints_after"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "stats_from_categories",
                "description": "Given BEFORE/AFTER categorical sequences, return counts, normalized distributions, competition ranks, Hellinger (0..1), Kendall's Tau-b (-1..1), and a disruption boolean per thresholds.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "labels_before": {"type": "array", "items": {"type": "string"}},
                        "labels_after": {"type": "array", "items": {"type": "string"}},
                        "domain": {"type": "array", "items": {"type": "string"}},
                        "thresholds": {
                            "type": "object",
                            "properties": {
                                "hellinger": {"type": "number", "default": 0.1},
                                "tau_b": {"type": "number", "default": 1.0}
                            }
                        }
                    },
                    "required": ["labels_before", "labels_after", "domain"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "cd_from_categories",
                "description": "Compute Category Diversity (CD) before/after and disruption flag with case-insensitive unique categories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "categories_before": {"type": "array", "items": {"type": "string"}},
                        "categories_after": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["categories_before", "categories_after"]
                }
            }
        },
    {
        "type": "function",
        "function": {
            "name": "categories_from_itinerary",
            "description": "Extract the category strings (2nd column) from an itinerary table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "itinerary": {
                        "type": "array",
                        "description": "Itinerary rows: [name, category, lon, lat, popularity]",
                        "items": {
                            "type": "array",
                            "items": {
                                # each cell may be string / number / boolean / null
                                "oneOf": [
                                    {"type": "string"},
                                    {"type": "number"},
                                    {"type": "boolean"},
                                    {"type": "null"}
                                ]
                            },
                            "minItems": 2
                        }
                    }
                },
                "required": ["itinerary"]
            }
        }
    }

]
