-- Author: Paul
-- Chronologically sample full resoultion training data by threshold; with feature engineered DC indicators
SELECT DISTINCT *
FROM `{{source_dataset}}.{{source_table}}`
ORDER BY start_time, end_time
