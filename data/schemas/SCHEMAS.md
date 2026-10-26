# Input schema contracts (UTF-8 CSV)

Every real data file MUST come from a documented authorized source. These CSVs are not distributed.

## EFCAMDAT / language proficiency
`learner_id,text,proficiency_level` — one row per essay. `proficiency_level`: integer 1..16 (unless original publisher's mapping establishes otherwise). The vocabulary tests and 16-week classroom outcomes do not come from EFCAMDAT.

## TOEFL task responses
`learner_id,task_id,task_type,response_time_seconds,item_difficulty,correct` — task_type in `reading,listening`, correct in `0,1`. The response records must be a separately licensed/collected learner dataset, NOT reconstructed from publicly available questions.

## MOOCCube processed behaviors
`learner_id,video_time,text_time,practice_count,mixed_resource_count` — values >= 0. MOOCCube raw export needs source-specific ETL; the repository provides validated analysis of this canonical behavioral extract, not a claim that MOOCCube directly contains these columns.

## Human-audited affect
`learner_id,task_id,text_score,audio_score,human_label` — numeric scores 0..1; human_label in negative/neutral/positive; audio_score can be blank when audio not recorded. Scores are outputs of separate, documented pretrained estimators; this repository does not claim to extract emotions from sound files without those estimators.

## Grammar errors
`learner_id,error_count,word_count,has_grammar_error` or supervised item-level training `learner_id,error_label,historical_error_rate,proficiency_score,item_complexity,transfer_risk`.

## 16-week classroom measures
One row per student/time point: `student_id,class_id,teacher_id,group,time,vocabulary,grammar_error_rate,fluency,speech_rate,pause_per_minute`.
`group` in `experimental,control`; `time` in `pre,post`. Four teachers each have one class of each group; 8 classes. Optional post-only fields in another CSV: `student_id,class_id,teacher_id,group,satisfaction,task_time_minutes,engagement_rate,delayed_retention_rate`.
The planned 240 students and 235 complete cases are article counts, NOT publicly distributable individual data.

## Six-condition task crossover
`student_id,condition,period,task_version,vocabulary,grammar_accuracy,fluency,task_satisfaction,task_time_minutes`.
`condition` in `full,no_profile,no_generation,no_path,generic_ai,equal_practice_digital`. Exactly one row per student×condition for a complete block. Six periods, six parallel task versions.

## Exercise quality ratings
`task_id,strategy,rater_id,grammatical_correctness,pedagogical_relevance,cefr_alignment,answer_validity` with `strategy` in `profile_guided,generic_ai,human`; binary ratings `0,1`. Five independent raters per task and strategy; 200 distinct task IDs shared by the three strategies. No teacher names or student identifiers should be committed.

## Source summary of figures 4–8
`data/public_summary/figures_4_8.csv` was transcribed from the user-supplied workbook. It contains only means without SDs or student/class labels. Figure 7 category IDs A/B/C/D must not be treated as verified dimension names; Figure 8 category IDs E/F/G are paired with metrics using the draft manuscript interpretation. The graphing script DOES NOT add invented error bars or significance stars.
