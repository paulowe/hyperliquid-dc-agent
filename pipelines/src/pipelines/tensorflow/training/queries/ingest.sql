-- Copyright 2022 Google LLC

-- Author: Paul C. Owe
-- Date: 21 June, 2025

 SELECT DISTINCT *
    FROM `{{project}}.{{dataset_id}}.{{table_id}}`
    WHERE threshold = {th}
    ORDER BY LOAD_TIME