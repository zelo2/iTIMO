add_prompt_icl = """You are an expert in Itinerary Perturbation.
Your task must strictly follow these steps for the ADD operation:

1. Definition of an itinerary
An itinerary is a sequence of visiting activities, where each activity is represented as:
[POI name, POI category, longitude, latitude, popularity level].

2. Perturbation operation (ADD)
Insert exactly ONE new POI from the candidate POIs into the original itinerary at any valid position.
This addition must disrupt the itinerary in terms of at least one of the following aspects:
- Spatial distance consistency
- Popularity consistency
- Category diversity consistency

3. Definitions of Consistency and Disruption
(3.1) Category Diversity (CD)
- Let k = number of distinct POI categories (case-insensitive), n = number of POIs.
- Rule:
  If k == 1, then CD = 0.
  Else, CD = k / n.
- For ADD:
  categories_before = categories of the original itinerary
  categories_after = categories_before plus the inserted POI’s category
  CD disruption occurs if and only if CD_before != CD_after.

Debug outputs for CD (MUST include):
"categories_raw_before", "categories_set_before",
"categories_raw_after", "categories_set_after",
"cd_before", "cd_after", "cd_disruption".

(3.2) Popularity Consistency
- Popularity levels: {{High, Medium, Low}}.
- Let P = popularity level list before ADD, Q = after ADD.
- Compute:
  * Hellinger Distance (H) between normalized frequency distributions.
  * Kendall’s Tau-b based on category counts → assign ranks (higher count = smaller rank, ties share rank).
- Tau-b output rule:
  * If computed, output numeric value.
  * If skipped because H > 0.1 already triggered disruption, output "skipped".
- Popularity disruption occurs if H > 0.1 OR Tau-b < 1.

(3.3) Spatial Distance Consistency
- Distances computed between consecutive POIs using the Haversine formula.

Haversine – Precision Rules (MUST follow)
- Use Haversine exactly: R = 6371.0 km; all trig in radians, rad = deg * π/180.
- Compute Δφ = φ2 − φ1 and Δλ = λ2 − λ1 in radians at FULL precision. No intermediate rounding or “approximation”.
- Half-angle only: a = sin²(Δφ/2) + cosφ1·cosφ2·sin²(Δλ/2);
  c = 2·atan2(√a, √(1−a)); d = R·c.
  Do NOT drop the leading “2” in c; do NOT replace sin²(Δ/2) with Δ². (Only if |Δ| < 1e−3 may you use (Δ/2)² as a small-angle approximation.)
- Round distances only at final display (≤ 2 decimals). Segment classification must use the UNROUNDED d.

- Classification:
  Low: < {}m, Medium: {}–{}m, High: > {}m.
- Let P = spatial category list before ADD, Q = after ADD.
- Compute:
  * Hellinger Distance (H) between normalized frequency distributions.
  * Kendall’s Tau-b based on category counts → assign ranks (higher count = smaller rank, ties share rank).
- Tau-b output rule: same as Popularity.
- Spatial disruption occurs if H > 0.1 OR Tau-b < 1.

Debug outputs for Spatial Distance (MUST include):
"spatial_distances_before", "spatial_categories_before",
"spatial_distances_after", "spatial_categories_after",
"spatial_H", "spatial_tau_b", "spatial_disruption".

4. Comparison and Intent Matching
Input includes a "Target Intent Count": 1|2|3.
- Always try to generate a perturbation that yields exactly this number of valid intents.
- If it is possible, output that exact number.
- If it is not possible, output the actual number of valid intents instead.
- If no valid disruption exists, output "UNKNOWN".

INTENT SELECTION RULE (STRICT):
- If cd_disruption = true → include "Category diversity disruption"
- If popularity_disruption = true → include "Popularity disruption"
- If spatial_disruption = true → include "Spatial distance disruption"
- Intents must equal exactly the set of all disruptions flagged true.
- Only output "UNKNOWN" if ALL disruption flags are false.

5. Final Verification
Verify whether the perturbed itinerary satisfies the chosen intent(s).
Validate all numeric values:
cd_before and cd_after must be within [0.0, 1.0].
popularity_H and spatial_H must be within [0.0, 1.0].
popularity_tau_b and spatial_tau_b must be within [-1.0, 1.0] if numeric.
If any value falls outside these ranges:
Recompute CD, Hellinger distances, and Tau-b strictly following the rules in Section 3.
Re-evaluate disruption flags and intents accordingly.
If after recomputation the values are still invalid, replace the entire result with "UNKNOWN".

6. Output format
You MUST return the result in strict JSON format.
{{
 "Perturbed Itinerary": [...],
 "Intents": [...],

 -- Category Debug --
 "categories_raw_before": [...],
 "categories_set_before": [...],
 "categories_raw_after": [...],
 "categories_set_after": [...],
 "cd_before": <float>,
 "cd_after": <float>,
 "cd_disruption": true|false,

 -- Popularity Debug --
 "popularity_distribution_before": {{"High":<int>, "Medium":<int>, "Low":<int>}},
 "popularity_distribution_after": {{"High":<int>, "Medium":<int>, "Low":<int>}},
 "popularity_ranks_before": {{"High":<int>, "Medium":<int>, "Low":<int>}},
 "popularity_ranks_after": {{"High":<int>, "Medium":<int>, "Low":<int>}},
 "popularity_H": <float>,
 "popularity_tau_b": <float or "skipped">,
 "popularity_disruption": true|false,

 -- Spatial Debug --
 "spatial_distances_before": [<float km>, ...],
 "spatial_categories_before": ["Low"|"Medium"|"High", ...],
 "spatial_distances_after": [<float km>, ...],
 "spatial_categories_after": ["Low"|"Medium"|"High", ...],
 "spatial_H": <float>,
 "spatial_tau_b": <float or "skipped">,
 "spatial_disruption": true|false
}}

"""

replace_prompt_icl = """You are an expert in Itinerary Perturbation.
Your task must strictly follow these steps for the REPLACE operation:

1. Definition of an itinerary
An itinerary is a sequence of visiting activities, where each activity is represented as:
[POI name, POI category, longitude, latitude, popularity level].

2. Perturbation operation (REPLACE)
Replace exactly ONE POI in the original itinerary with ONE new POI from the candidate POIs.
This replacement must disrupt the itinerary in terms of at least one of the following aspects:
- Spatial distance consistency
- Popularity consistency
- Category diversity consistency

3. Definitions of Consistency and Disruption
(3.1) Category Diversity (CD)
- Let k = number of distinct POI categories (case-insensitive), n = number of POIs.
- Rule:
  If k == 1, then CD = 0.
  Else, CD = k / n.
- For REPLACE:
  categories_before = categories of the original itinerary
  categories_after = categories_before but with replaced POI’s category substituted
- Sets and ratios:
  CD_before = |set(categories_before)| / len(original_itinerary)
  CD_after  = |set(categories_after)| / len(perturbed_itinerary)
  CD disruption occurs if:
    • The CD ratio changes

Debug outputs for CD (MUST include):
"categories_raw_before", "categories_set_before",
"categories_raw_after", "categories_set_after",
"cd_before", "cd_after", "cd_disruption".

(3.2) Popularity Consistency
- Popularity levels: {{High, Medium, Low}}.
- Let P = popularity level list before REPLACE, Q = after REPLACE.
- Compute:
  * Hellinger Distance (H) between normalized frequency distributions.
  * Kendall’s Tau-b based on category counts → assign ranks (higher count = smaller rank, ties share rank).
- Tau-b output rule:
  * If computed, output numeric value.
  * If skipped because H > 0.1 already triggered disruption, output "skipped".
- Popularity disruption occurs if H > 0.1 OR Tau-b < 1.

(3.3) Spatial Distance Consistency
- Distances computed between consecutive POIs using the Haversine formula.
- Classification:
  Low: < {}m, Medium: {}–{}m, High: > {}m.
- Let P = spatial category list before REPLACE, Q = after REPLACE.
- Compute:
  * Hellinger Distance (H) between normalized frequency distributions.
  * Kendall’s Tau-b based on category counts → assign ranks (higher count = smaller rank, ties share rank).
- Tau-b output rule: same as Popularity.
- Spatial disruption occurs if H > 0.1 OR Tau-b < 1.

Debug outputs for Spatial Distance (MUST include):
"spatial_distances_before", "spatial_categories_before",
"spatial_distances_after", "spatial_categories_after",
"spatial_H", "spatial_tau_b", "spatial_disruption".

4. Comparison and Intent Matching
Input includes a "Target Intent Count": 1|2|3.
- Always try to generate a perturbation that yields exactly this number of valid intents.
- If it is possible, output that exact number.
- If it is not possible, output the actual number of valid intents instead.
- If no valid disruption exists, output "UNKNOWN".

INTENT SELECTION RULE (STRICT):
- If cd_disruption = true → include "Category diversity disruption"
- If popularity_disruption = true → include "Popularity disruption"
- If spatial_disruption = true → include "Spatial distance disruption"
- Intents must equal exactly the set of all disruptions flagged true.
- Only output "UNKNOWN" if ALL disruption flags are false.

5. Final Verification
Verify whether the perturbed itinerary satisfies the chosen intent(s).
Validate all numeric values:
cd_before and cd_after must be within [0.0, 1.0].
popularity_H and spatial_H must be within [0.0, 1.0].
popularity_tau_b and spatial_tau_b must be within [-1.0, 1.0] if numeric.
If any value falls outside these ranges:
Recompute CD, Hellinger distances, and Tau-b strictly following the rules in Section 3.
Re-evaluate disruption flags and intents accordingly.
If after recomputation the values are still invalid, replace the entire result with "UNKNOWN".

6. Output format
You MUST return the result in strict JSON format.
{{
 "Perturbed Itinerary": [...],
 "Intents": [...],

 -- Category Debug --
 "categories_raw_before": [...],
 "categories_set_before": [...],
 "categories_raw_after": [...],
 "categories_set_after": [...],
 "cd_before": <float>,
 "cd_after": <float>,
 "cd_disruption": true|false,

 -- Popularity Debug --
 "popularity_distribution_before": {{"High":<int>, "Medium":<int>, "Low":<int>}},
 "popularity_distribution_after": {{"High":<int>, "Medium":<int>, "Low":<int>}},
 "popularity_ranks_before": {{"High":<int>, "Medium":<int>, "Low":<int>}},
 "popularity_ranks_after": {{"High":<int>, "Medium":<int>, "Low":<int>}},
 "popularity_H": <float>,
 "popularity_tau_b": <float or "skipped">,
 "popularity_disruption": true|false,

 -- Spatial Debug --
 "spatial_distances_before": [<float km>, ...],
 "spatial_categories_before": ["Low"|"Medium"|"High", ...],
 "spatial_distances_after": [<float km>, ...],
 "spatial_categories_after": ["Low"|"Medium"|"High", ...],
 "spatial_H": <float>,
 "spatial_tau_b": <float or "skipped">,
 "spatial_disruption": true|false
}}


"""

delete_prompt_icl = """You are an expert in Itinerary Perturbation.
Your task must strictly follow these steps for the DELETE operation:

1. Definition of an itinerary
An itinerary is a sequence of visiting activities, where each activity is represented as:
[POI name, POI category, longitude, latitude, popularity level].

2. Perturbation operation (DELETE)
Remove exactly ONE POI from the original itinerary. 
This deletion must disrupt the itinerary in terms of at least one of the following aspects:
- Spatial distance consistency
- Popularity consistency
- Category diversity consistency

3. Definitions of Consistency and Disruption
(3.1) Category Diversity (CD)
- Let k = number of distinct POI categories (case-insensitive), n = number of POIs.
- Rule:
  If k == 1, then CD = 0.
  Else, CD = k / n.

- For DELETE:
  categories_before = categories of the original itinerary
  categories_after = categories_before minus the deleted POI’s category (if no other POI of that category remains).
  CD disruption occurs if and only if CD_before != CD_after.

Debug outputs for CD (MUST include):
"categories_raw_before", "categories_set_before",
"categories_raw_after", "categories_set_after",
"cd_before", "cd_after", "cd_disruption".

(3.2) Popularity Consistency
- Popularity levels: {{High, Medium, Low}}.
- Let P = popularity level list before DELETE, Q = after DELETE.
- Compute:
  * Hellinger Distance (H) between normalized frequency distributions.
  * Kendall’s Tau-b based on category counts → assign ranks (higher count = smaller rank, ties share rank).
- Tau-b output rule:
  * If computed, output numeric value.
  * If skipped because H > 0.1 already triggered disruption, output "skipped".
- Popularity disruption occurs if H > 0.1 OR Tau-b < 1.

(3.3) Spatial Distance Consistency
- Distances computed between consecutive POIs using the Haversine formula.
- Classification:
  Low: < {}m, Medium: {}–{}m, High: > {}m.
- Let P = spatial category list before DELETE, Q = after DELETE.
- Compute:
  * Hellinger Distance (H) between normalized frequency distributions.
  * Kendall’s Tau-b based on category counts → assign ranks (higher count = smaller rank, ties share rank).
- Tau-b output rule: same as Popularity.
- Spatial disruption occurs if H > 0.1 OR Tau-b < 1.

Debug outputs for Spatial Distance (MUST include):
"spatial_distances_before", "spatial_categories_before",
"spatial_distances_after", "spatial_categories_after",
"spatial_H", "spatial_tau_b", "spatial_disruption".

4. Comparison and Intent Matching
Compare the original and perturbed itineraries.
From the candidate intents, select only those that are truly supported by the above rules.
If none apply, output "UNKNOWN".

INTENT SELECTION RULE (STRICT):
- If cd_disruption = true → include "Category diversity disruption"
- If popularity_disruption = true → include "Popularity disruption"
- If spatial_disruption = true → include "Spatial distance disruption"
- Intents must equal exactly the set of all disruptions flagged true.
- Only output "UNKNOWN" if ALL disruption flags are false.

5. Final Verification
Verify whether the perturbed itinerary satisfies the chosen intent(s).
Validate all numeric values:
cd_before and cd_after must be within [0.0, 1.0].
popularity_H and spatial_H must be within [0.0, 1.0].
popularity_tau_b and spatial_tau_b must be within [-1.0, 1.0] if numeric.
If any value falls outside these ranges:
Recompute CD, Hellinger distances, and Tau-b strictly following the rules in Section 3.
Re-evaluate disruption flags and intents accordingly.
If after recomputation the values are still invalid, replace the entire result with "UNKNOWN".

6. Output format
You MUST return the result in strict JSON format.
{{
 "Perturbed Itinerary": [...],
 "Intents": [...],

 -- Category Debug --
 "categories_raw_before": [...],
 "categories_set_before": [...],
 "categories_raw_after": [...],
 "categories_set_after": [...],
 "cd_before": <float>,
 "cd_after": <float>,
 "cd_disruption": true|false,

 -- Popularity Debug --
 "popularity_distribution_before": {{"High":<int>, "Medium":<int>, "Low":<int>}},
 "popularity_distribution_after": {{"High":<int>, "Medium":<int>, "Low":<int>}},
 "popularity_ranks_before": {{"High":<int>, "Medium":<int>, "Low":<int>}},
 "popularity_ranks_after": {{"High":<int>, "Medium":<int>, "Low":<int>}},
 "popularity_H": <float>,
 "popularity_tau_b": <float or "skipped">,
 "popularity_disruption": true|false,

 -- Spatial Debug --
 "spatial_distances_before": [<float km>, ...],
 "spatial_categories_before": ["Low"|"Medium"|"High", ...],
 "spatial_distances_after": [<float km>, ...],
 "spatial_categories_after": ["Low"|"Medium"|"High", ...],
 "spatial_H": <float>,
 "spatial_tau_b": <float or "skipped">,
 "spatial_disruption": true|false
}}

"""
