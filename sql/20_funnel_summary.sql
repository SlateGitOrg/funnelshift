-- Per cohort week and segment: users reaching each step, step and cumulative conversion, and
-- the median and p90 time from the previous step.
--
-- Input : funnel_steps, funnel_reach, funnel_step_defs (from 10_funnel_steps.sql.j2),
--         funnel_params(as_of, window_days)
-- Output: funnel_cohort_status  one row per cohort week and whether its window has elapsed
--         funnel_summary        only cohorts whose full window has elapsed
--
-- A cohort week is complete when its last possible signup (the end of the week) plus the
-- window is not after `as_of`, the latest event time in the data. Younger cohorts would count
-- late converters as non-converters, so they are not reported.

CREATE OR REPLACE TABLE funnel_cohort_status AS
SELECT s.cohort_week,
       s.cohort_start,
       count(*) AS users,
       s.cohort_start + INTERVAL 7 DAY + to_days(CAST(p.window_days AS INTEGER)) <= p.as_of
           AS complete
FROM funnel_steps AS s
CROSS JOIN funnel_params AS p
GROUP BY s.cohort_week, s.cohort_start, p.as_of, p.window_days;

CREATE OR REPLACE TABLE funnel_summary AS
WITH complete_cohorts AS (
    SELECT cohort_week FROM funnel_cohort_status WHERE complete
),
reached AS (
    SELECT s.cohort_week,
           s.cohort_start,
           s.segment,
           d.step_no,
           d.step_name,
           count(*) FILTER (WHERE s.furthest_step >= d.step_no) AS users_reached
    FROM funnel_steps AS s
    JOIN complete_cohorts AS c ON c.cohort_week = s.cohort_week
    CROSS JOIN funnel_step_defs AS d
    GROUP BY s.cohort_week, s.cohort_start, s.segment, d.step_no, d.step_name
),
conversion AS (
    SELECT *,
           CASE WHEN step_no = 1 THEN NULL
                ELSE CAST(users_reached AS DOUBLE)
                     / nullif(lag(users_reached) OVER w, 0)
           END AS step_conv,
           CAST(users_reached AS DOUBLE) / nullif(first_value(users_reached) OVER w, 0)
               AS cum_conv
    FROM reached
    WINDOW w AS (
        PARTITION BY cohort_week, segment ORDER BY step_no
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )
),
gaps AS (
    SELECT cohort_week,
           segment,
           step_no,
           gap_secs
    FROM (
        SELECT s.cohort_week,
               s.segment,
               r.step_no,
               epoch(r.step_ts) - epoch(lag(r.step_ts) OVER (
                   PARTITION BY r.user_id ORDER BY r.step_no
               )) AS gap_secs
        FROM funnel_reach AS r
        JOIN funnel_steps AS s ON s.user_id = r.user_id
        JOIN complete_cohorts AS c ON c.cohort_week = s.cohort_week
    )
    WHERE step_no > 1
),
gap_stats AS (
    SELECT cohort_week,
           segment,
           step_no,
           quantile_cont(gap_secs, 0.5) AS median_secs,
           quantile_cont(gap_secs, 0.9) AS p90_secs
    FROM gaps
    GROUP BY cohort_week, segment, step_no
)
SELECT c.cohort_week,
       c.cohort_start,
       c.segment,
       c.step_no,
       c.step_name,
       c.users_reached,
       c.step_conv,
       c.cum_conv,
       g.median_secs,
       g.p90_secs
FROM conversion AS c
LEFT JOIN gap_stats AS g
  ON g.cohort_week = c.cohort_week AND g.segment = c.segment AND g.step_no = c.step_no
ORDER BY c.cohort_week, c.segment, c.step_no;
