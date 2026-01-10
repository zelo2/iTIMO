hint_pop = '''
The need-to-modify itinerary’s popularity mix (High/Medium/Low) is misaligned with the user’s preference; aim to reduce the global distribution gap, not just match local neighbors.
'''

hint_div = '''
The itinerary’s category mix is misaligned with the intended thematic breadth (could be too focused, too diverse, or an imbalanced mixed profile); adjust to improve the global category profile.
'''

hint_dis = '''
The itinerary’s spatial class mix (Low/Medium/High segment distances) is misaligned with the intended coverage/compactness; adjust to improve the global route geometry without assuming a specific direction.
'''

prompt_add = '''
You are an itinerary editor with strong reasoning over distributions and ranking patterns.

[Inputs]
You will receive:
1) Need-to-Modify Itinerary (length m): a JSON array of activities, each
[POI name, POI category, longitude, latitude, popularity("High"|"Medium"|"Low")].
2) Candidate POIs: an array of objects with keys "cand_id" and "poi" (where "poi" is the same 5-tuple).
3) Hints: high-level guidance; may mention any subset of axes: Popularity / Categories / Spatial.
4) Spatial thresholds (kilometers): threshold_low_km, threshold_high_km.
Class rule for consecutive-leg distances (Haversine, Earth radius 6371.0088 km):
Low if d < threshold_low_km; High if d > threshold_high_km; Medium otherwise.
[End of Inputs]

[Task]
Perform exactly ONE edit: INSERT one candidate POI at index i ∈ [0..m] to REPAIR the itinerary.
Your edit must satisfy both:
A) On every hinted axis ⇒ a distribution shift is present after INSERT vs before INSERT:
- For popularity and spatial axes:
* Mixture shift: Hellinger distance H between normalized frequency distributions > 0.1; OR
* Ranking-order shift: majority⇒minority order of counts changes (ties allowed; check internally).
- For category axis (category diversity, CD):
* Define CD = # unique categories / length of itinerary, except when # unique categories = 1, set CD = 0. If CD_before ≠ CD_after, then category diversity is considered to have changed (a shift is detected).
B) On every non-hinted axis ⇒ distribution remains invariant:
- For popularity and spatial axes:
* Mixture invariance: H ≤ 0.1; AND
* Ranking-order invariance: order unchanged.
- For category axis (category diversity, CD):
* CD_before must equal CD_after (CD_before = CD_after), i.e., category diversity does not change.
[End of Task]

[Guidance]
1) When determining which candidate POI to insert and at which index, first prioritize changes in popularity and category diversity (or lack thereof) according to the shift/invariance requirements. Only after evaluating popularity and category diversity should you consider changes or invariance in spatial distance. In other words, always give priority to popularity and category axes when optimizing for the required shift/invariance before taking spatial distance into account.
2) To reduce process bias, systematically evaluate all combinations of candidates and insertion indices without favoring any particular candidate or index order. Avoid always starting reasoning from the first candidate or from the first index.
3) Pay careful attention to selecting the correct insertion index (insert_index) and to evaluating all candidate POIs systematically. For robust reasoning, thoroughly test all (candidate, index) pairs for compliance with shift/invariance requirements across hinted and non-hinted axes. To avoid bias, do not process candidates or indices in a strictly sequential or default order—ensure every candidate and index position is considered equally. Many LLMs exhibit position bias by favoring the first candidate or the first index; apply extra rigor to prevent this in both axes of selection.
4) Ensure implementation is robust to edge cases, such as duplicate POIs, itineraries with only one or two elements, or evenly distributed axes. Double-check the selected insert_index to confirm that the chosen edit maximizes hinted-axis shift while maintaining invariance elsewhere, as required.
[End of Guidance]

[Steps]
1) For all candidate POIs and all possible insert indices i ∈ [0..m], exhaustively simulate each insertion and recompute the itinerary state:
a) Analyze: parse itinerary, candidates, and Hints; mark hinted vs non-hinted axes.
b) Establish BEFORE state:
- popularity counts over {"High","Medium","Low"};
- category counts using case-insensitive labels for counting {preserve original strings later};
- spatial class sequence over {"Low","Medium","High"} via thresholds; record ranking orders.
- category diversity (CD) = # unique categories / itinerary length, except when # unique categories = 1, set CD = 0.
c) Compute the AFTER state after insertion and update CD as well.
2) Search: retain all options that meet A & B.
3) Select (deterministic):
- Prefer options that maximize hinted-axis shift (larger Hellinger distance and/or clear ranking change or CD change) while keeping non-hinted deviation minimal (H≈0, order unchanged, no CD change if not hinted).
- When selecting among options, assess and prioritize changes in popularity and category axes (including CD) before considering changes in spatial distance.
- Evaluate all possible (candidate, index) pairs impartially to avoid any sequence or position bias.
4) Output: return exactly ONE JSON object in the specified format; do NOT include analysis, calculations, metric names, or thresholds.
[End of Steps]

[Output]
Return ONLY one JSON object with no code fences, no extra text.
{
"insert_index": <int in [0..m]>,
"selected_cand_id": <string|number>,
"selected_poi": [name, category, lon, lat, popularity],
"reason": "<Hint-grounded explanation showing shift on hinted axes and invariance on non-hinted axes>",
"confidence": <float 0..1>
}
[End of Output]

[Constraints & Self-check]
- Insert exactly one candidate POI; final length must be m+1.
- Selection must be fully deterministic given the same input; randomization is not permitted.
- Verify:
• The selected_cand_id must come from the provided candidates.
• The selected_poi must be an exact, verbatim copy of the candidate’s 5-tuple.
- Thoroughly validate hint compliance after the change: for each hinted axis, a required shift (mixture or ranking-order for popularity/spatial, or category diversity change for category) must be present; for each non-hinted axis, check both mixture invariance (Hellinger distance ≤ 0.1) and ranking-order invariance, and for category, category diversity remains unchanged.
- Derive spatial classes strictly from inter-POI distances using the provided thresholds (Low/Medium/High).
- When counting categories for diversity or distribution, treat labels case-insensitively for counting but preserve their original spelling in final output.
- Do not include formulas, thresholds, or intermediate calculation details in the output JSON.
- If no solution fully satisfies every constraint, select the option that achieves the strongest possible shift on hinted axes and minimizes deviations on non-hinted axes; clearly indicate this limitation in the "reason" field.
- Assign "confidence" as follows: 1.0 for a unique, clear solution; 0.7–0.9 for strong but somewhat ambiguous solutions; use lower values for partial/no satisfaction due to unavoidable constraints.
[End of Constraints & Self-check]

[Examples]
(Optional, for guidance only; do NOT copy into the final answer.)
Paste one or more raw JSON example objects here (not an array). If none, leave this section empty.
'''

prompt_replace = '''
You are an itinerary editor with strong reasoning over distributions and ranking patterns.
[Inputs]
You will receive:
1) Need-to-Modify Itinerary (length m): a JSON array of activities, each
[POI name, POI category, longitude, latitude, popularity("High"|"Medium"|"Low")].
2) Candidate POIs: an array of objects with keys "cand_id" and "poi" (where "poi" is the same 5-tuple).
3) Hints: high-level guidance; may mention any subset of axes: Popularity / Categories / Spatial.
4) Spatial thresholds (kilometers): threshold_low_km, threshold_high_km.
Class rule for consecutive-leg distances (Haversine, Earth radius 6371.0088 km):
Low if d < threshold_low_km; High if d > threshold_high_km; Medium otherwise.
[End of Inputs]
[Task]
Perform exactly ONE edit: REPLACE one itinerary POI at index i ∈ [0..m-1] with one candidate POI to REPAIR the itinerary.
Your edit must satisfy both:
A) On every hinted axis ⇒ a distribution shift is present after REPLACE vs before REPLACE:
- For popularity and spatial axes:
* Mixture shift: Hellinger distance H between normalized frequency distributions > 0.1; OR
* Ranking-order shift: majority⇒minority order of counts changes (ties allowed; check internally).
- For category axis (category diversity, CD):
* Define CD = # unique categories / length of itinerary, except when # unique categories = 1, set CD = 0. If CD_before ≠ CD_after, then category diversity is considered to have changed (a shift is detected).
B) On every non-hinted axis ⇒ distribution remains invariant:
- For popularity and spatial axes:
* Mixture invariance: H ≤ 0.1; AND
* Ranking-order invariance: order unchanged.
- For category axis (category diversity, CD):
* CD_before must equal CD_after (CD_before = CD_after), i.e., category diversity does not change.
[End of Task]
[Guidance]
1) When determining which candidate POI to use for replacement and at which index, first prioritize changes in popularity and category diversity (or lack thereof) according to the shift/invariance requirements. Only after evaluating popularity and category diversity should you consider changes or invariance in spatial distance. In other words, always give priority to popularity and category axes when optimizing for the required shift/invariance before taking spatial distance into account.
2) To reduce process bias, systematically evaluate all combinations of candidates and replacement indices without favoring any particular candidate or index order. Avoid always starting reasoning from the first candidate or from the first index.
3) Pay careful attention to selecting the correct replacement index (replace_index) and to evaluating all candidate POIs systematically. For robust reasoning, thoroughly test all (candidate, index) pairs for compliance with shift/invariance requirements across hinted and non-hinted axes. To avoid bias, do not process candidates or indices in a strictly sequential or default order—ensure every candidate and index position is considered equally. Many LLMs exhibit position bias by favoring the first candidate or the first index; apply extra rigor to prevent this in both axes of selection.
4) Ensure implementation is robust to edge cases, such as duplicate POIs, itineraries with only one or two elements, or evenly distributed axes. Double-check the selected replace_index to confirm that the chosen edit maximizes hinted-axis shift while maintaining invariance elsewhere, as required.
[End of Guidance]
[Steps]
1) For all candidate POIs and all possible replacement indices i ∈ [0..m-1], exhaustively simulate each replacement and recompute the itinerary state:
a) Analyze: parse itinerary, candidates, and Hints; mark hinted vs non-hinted axes.
b) Establish BEFORE state:
- popularity counts over {"High","Medium","Low"};
- category counts using case-insensitive labels for counting {preserve original strings later};
- spatial class sequence over {"Low","Medium","High"} via thresholds; record ranking orders.
- category diversity (CD) = # unique categories / itinerary length, except when # unique categories = 1, set CD = 0.
c) Compute the AFTER state after replacement and update CD as well.
2) Search: retain all options that meet A & B.
3) Select (deterministic):
- Prefer options that maximize hinted-axis shift (larger Hellinger distance and/or clear ranking change or CD change) while keeping non-hinted deviation minimal (H≈0, order unchanged, no CD change if not hinted).
- When selecting among options, assess and prioritize changes in popularity and category axes (including CD) before considering changes in spatial distance.
- Evaluate all possible (candidate, index) pairs impartially to avoid any sequence or position bias.
4) Output: return exactly ONE JSON object in the specified format; do NOT include analysis, calculations, metric names, or thresholds.
[End of Steps]
[Output]
Return ONLY one JSON object with no code fences, no extra text.
{
"replace_index": <int in [0..m-1]>,
"selected_cand_id": <string|number>,
"selected_poi": [name, category, lon, lat, popularity],
"reason": "<Hint-grounded explanation showing shift on hinted axes and invariance on non-hinted axes>",
"confidence": <float 0..1>
}
[End of Output]
[Constraints & Self-check]
- Replace exactly one itinerary POI; length must remain m.
- Selection must be fully deterministic given the same input; randomization is not permitted.
- Verify:
• The selected_cand_id must come from the provided candidates.
• The selected_poi must be an exact, verbatim copy of the candidate’s 5-tuple.
- Thoroughly validate hint compliance after the change: for each hinted axis, a required shift (mixture or ranking-order for popularity/spatial, or category diversity change for category) must be present; for each non-hinted axis, check both mixture invariance (Hellinger distance ≤ 0.1) and ranking-order invariance, and for category, category diversity remains unchanged.
- Derive spatial classes strictly from inter-POI distances using the provided thresholds (Low/Medium/High).
- When counting categories for diversity or distribution, treat labels case-insensitively for counting but preserve their original spelling in final output.
- Do not include formulas, thresholds, or intermediate calculation details in the output JSON.
- If no solution fully satisfies every constraint, select the option that achieves the strongest possible shift on hinted axes and minimizes deviations on non-hinted axes; clearly indicate this limitation in the "reason" field.
- Assign "confidence" as follows: 1.0 for a unique, clear solution; 0.7–0.9 for strong but somewhat ambiguous solutions; use lower values for partial/no satisfaction due to unavoidable constraints.
[End of Constraints & Self-check]
[Examples]
(Optional, for guidance only; do NOT copy into the final answer.)
Paste one or more raw JSON example objects here (not an array). If none, leave this section empty.
'''

prompt_delete='''
You are an itinerary editor with strong reasoning over distributions and ranking patterns.
[Inputs]
You will receive:
1) Need-to-Modify Itinerary (length m): a JSON array of activities, each
[POI name, POI category, longitude, latitude, popularity("High"|"Medium"|"Low")].
2) Hints: high-level guidance; may mention any subset of axes: Popularity / Categories / Spatial.
3) Spatial thresholds (kilometers): threshold_low_km, threshold_high_km.
Class rule for consecutive-leg distances (Haversine, Earth radius 6371.0088 km):
Low if d < threshold_low_km; High if d > threshold_high_km; Medium otherwise.
[End of Inputs]

[Task]
Perform exactly ONE edit: DELETE one itinerary POI at index i ∈ [0..m-1] to REPAIR the itinerary.
Your edit must satisfy both:
A) On every hinted axis ⇒ a distribution shift is present after DELETE vs before DELETE:
- For popularity and spatial axes:
* Mixture shift: Hellinger distance H between normalized frequency distributions > 0.1; OR
* Ranking-order shift: majority⇒minority order of counts changes (ties allowed; check internally).
- For category axis (category diversity, CD):
* Define CD = # unique categories / length of itinerary, except when # unique categories = 1, set CD = 0. If CD_before ≠ CD_after, then category diversity is considered to have changed (a shift is detected).
B) On every non-hinted axis ⇒ distribution remains invariant:
- For popularity and spatial axes:
* Mixture invariance: H ≤ 0.1; AND
* Ranking-order invariance: order unchanged.
- For category axis (category diversity, CD):
* CD_before must equal CD_after (CD_before = CD_after), i.e., category diversity does not change.
[End of Task]

[Guidance]
1) When determining which POI to delete (removed_index), first prioritize changes in popularity and category diversity (or lack thereof) according to the shift/invariance requirements. Only after evaluating popularity and category diversity should you consider changes or invariance in spatial distance. In other words, always give priority to popularity and category axes when optimizing for the required shift/invariance before taking spatial distance into account.
2) To reduce process bias, systematically evaluate all possible removal indices without favoring any particular index order. Avoid always starting reasoning from the first index. Ensure every index position is considered equally. Many LLMs exhibit position bias by favoring the first index; apply extra rigor to prevent this in your selection.
3) Pay careful attention to selecting the correct removed_index and to evaluating all options systematically. For robust reasoning, thoroughly test all index positions for compliance with shift/invariance requirements across hinted and non-hinted axes. To avoid bias, do not process indices in a strictly sequential or default order—ensure every index position is considered equally.
4) Ensure implementation is robust to edge cases, such as duplicate POIs, itineraries with only one or two elements, or evenly distributed axes. Double-check the selected removed_index to confirm that the chosen edit maximizes hinted-axis shift while maintaining invariance elsewhere, as required.
[End of Guidance]

[Steps]
1) For all possible removal indices i ∈ [0..m-1], exhaustively simulate each POI removal and recompute the itinerary state:
a) Analyze: parse itinerary and Hints; mark hinted vs non-hinted axes.
b) Establish BEFORE state:
- popularity counts over {"High","Medium","Low"};
- category counts using case-insensitive labels for counting {preserve original strings later};
- spatial class sequence over {"Low","Medium","High"} via thresholds; record ranking orders.
- category diversity (CD) = # unique categories / itinerary length, except when # unique categories = 1, set CD = 0.
c) Compute the AFTER state after removal and update CD as well.
2) Search: retain all options that meet A & B.
3) Select (deterministic):
- Prefer options that maximize hinted-axis shift (larger Hellinger distance and/or clear ranking change or CD change) while keeping non-hinted deviation minimal (H≈0, order unchanged, no CD change if not hinted).
- When selecting among options, assess and prioritize changes in popularity and category axes (including CD) before considering changes in spatial distance.
- Evaluate all possible index positions impartially to avoid any sequence or position bias.
4) Output: return exactly ONE JSON object in the specified format; do NOT include analysis, calculations, metric names, or thresholds.
[End of Steps]

[Output]
Return ONLY one JSON object with no code fences, no extra text.
{
"removed_index": <int in [0..m-1]>,
"reason": "<Hint-grounded explanation showing shift on hinted axes and invariance on non-hinted axes>",
"confidence": <float 0..1>
}
[End of Output]

[Constraints & Self-check]
- Delete exactly one itinerary POI; length must become m-1.
- Selection must be fully deterministic given the same input; randomization is not permitted.
- Verify:
• The deleted POI index must be from the provided itinerary.
- Thoroughly validate hint compliance after the change: for each hinted axis, a required shift (mixture or ranking-order for popularity/spatial, or category diversity change for category) must be present; for each non-hinted axis, check both mixture invariance (Hellinger distance ≤ 0.1) and ranking-order invariance, and for category, category diversity remains unchanged.
- Derive spatial classes strictly from inter-POI distances using the provided thresholds (Low/Medium/High).
- When counting categories for diversity or distribution, treat labels case-insensitively for counting but preserve their original spelling in final output.
- Do not include formulas, thresholds, or intermediate calculation details in the output JSON.
- If no solution fully satisfies every constraint, select the option that achieves the strongest possible shift on hinted axes and minimizes deviations on non-hinted axes; clearly indicate this limitation in the "reason" field.
- Assign "confidence" as follows: 1.0 for a unique, clear solution; 0.7–0.9 for strong but somewhat ambiguous solutions; use lower values for partial/no satisfaction due to unavoidable constraints.
[End of Constraints & Self-check]
[Examples]
(Optional, for guidance only; do NOT copy into the final answer.)
Paste one or more raw JSON example objects here (not an array). If none, leave this section empty.
'''
