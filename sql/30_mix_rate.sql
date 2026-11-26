-- Symmetric shift-share decomposition of the change in conversion between two periods.
--
-- For segment cells s with entry weights w (share of the period's entering users) and
-- conversion rates c, between a base period 0 and a test period 1:
--
--   total = sum(w1 * c1) - sum(w0 * c0)
--   mix   = sum((w1 - w0) * (c0 + c1) / 2)    effect of the change in who enters
--   rate  = sum((c1 - c0) * (w0 + w1) / 2)    effect of the change in each cell's rate
--
-- Midpoint weights make mix + rate equal total identically: there is no interaction term left
-- over. A cell with no users in one period borrows the other period's rate, so its whole
-- effect is mix. The same decomposition is applied to every step's conditional rate
-- (users at step k / users at step k-1), weighted by each cell's share of the users at step
-- k-1, which places the rate effect in a step x segment cell.
--
-- The per-cell mix term is not centred: a cell that gains weight shows a positive term
-- whenever its conversion is positive, whatever it is relative to the average. The tables
-- therefore also carry mix_c = (w1 - w0) * ((c0 + c1) / 2 - Cbar), with Cbar the overall
-- midpoint conversion. It sums to the same total because the weight changes sum to zero, and
-- its sign says whether the cell pulled the average up or down. The step tables do the same
-- with the midpoint conditional rate.
--
-- Input : funnel_summary, funnel_step_defs, mixrate_weeks(week, period): the explicit ISO
--         week labels of period 0 (base) and period 1 (test)
-- Output: mix_rate_total, mix_rate_cells (end to end), mix_rate_steps (per step)

CREATE OR REPLACE TABLE mix_input AS
SELECT segment, step_no, period, sum(users_reached) AS users
FROM (
    SELECT s.segment,
           s.step_no,
           s.users_reached,
           w.period
    FROM funnel_summary AS s
    JOIN mixrate_weeks AS w ON w.week = s.cohort_week
)
GROUP BY segment, step_no, period;

CREATE OR REPLACE TABLE mix_grid AS
SELECT sg.segment,
       pr.period,
       d.step_no,
       d.step_name,
       coalesce(i.users, 0) AS users
FROM (SELECT DISTINCT segment FROM mix_input) AS sg
CROSS JOIN (VALUES (0), (1)) AS pr(period)
CROSS JOIN funnel_step_defs AS d
LEFT JOIN mix_input AS i
  ON i.segment = sg.segment AND i.period = pr.period AND i.step_no = d.step_no;

CREATE OR REPLACE TABLE mix_agg AS
SELECT segment,
       period,
       CAST(sum(users) FILTER (WHERE step_no = 1) AS DOUBLE) AS n_entry,
       CAST(sum(users) FILTER (WHERE step_no = (SELECT max(step_no) FROM funnel_step_defs))
            AS DOUBLE) AS n_last
FROM mix_grid
GROUP BY segment, period;

CREATE OR REPLACE TABLE mix_rate_cells AS
WITH wide AS (
    SELECT a0.segment,
           a0.n_entry AS n0,
           a1.n_entry AS n1,
           a0.n_entry / nullif(t0.total, 0) AS w0,
           a1.n_entry / nullif(t1.total, 0) AS w1,
           a0.n_last / nullif(a0.n_entry, 0) AS c0_raw,
           a1.n_last / nullif(a1.n_entry, 0) AS c1_raw
    FROM mix_agg AS a0
    JOIN mix_agg AS a1 ON a1.segment = a0.segment AND a0.period = 0 AND a1.period = 1
    CROSS JOIN (SELECT sum(n_entry) AS total FROM mix_agg WHERE period = 0) AS t0
    CROSS JOIN (SELECT sum(n_entry) AS total FROM mix_agg WHERE period = 1) AS t1
),
filled AS (
    SELECT segment, n0, n1,
           coalesce(w0, 0) AS w0,
           coalesce(w1, 0) AS w1,
           coalesce(c0_raw, c1_raw, 0) AS c0,
           coalesce(c1_raw, c0_raw, 0) AS c1
    FROM wide
)
SELECT segment, n0, n1, w0, w1, c0, c1,
       (w1 - w0) * (c0 + c1) / 2 AS mix,
       (w1 - w0) * ((c0 + c1) / 2
           - (sum(w0 * c0) OVER () + sum(w1 * c1) OVER ()) / 2) AS mix_c,
       (c1 - c0) * (w0 + w1) / 2 AS rate
FROM filled
ORDER BY segment;

CREATE OR REPLACE TABLE mix_rate_total AS
SELECT
    (SELECT sum(n_last) / nullif(sum(n_entry), 0) FROM mix_agg WHERE period = 0) AS conv_base,
    (SELECT sum(n_last) / nullif(sum(n_entry), 0) FROM mix_agg WHERE period = 1) AS conv_test,
    (SELECT sum(n_last) / nullif(sum(n_entry), 0) FROM mix_agg WHERE period = 1)
      - (SELECT sum(n_last) / nullif(sum(n_entry), 0) FROM mix_agg WHERE period = 0) AS delta,
    sum(mix) AS mix,
    sum(rate) AS rate,
    sum(n0) AS users_base,
    sum(n1) AS users_test
FROM mix_rate_cells;

CREATE OR REPLACE TABLE mix_rate_steps AS
WITH pairs AS (
    SELECT g.segment,
           g.period,
           g.step_no,
           g.step_name,
           CAST(g.users AS DOUBLE) AS n_k,
           CAST(p.users AS DOUBLE) AS n_prev
    FROM mix_grid AS g
    JOIN mix_grid AS p
      ON p.segment = g.segment AND p.period = g.period AND p.step_no = g.step_no - 1
),
shares AS (
    SELECT *,
           n_prev / nullif(sum(n_prev) OVER (PARTITION BY period, step_no), 0) AS v,
           n_k / nullif(n_prev, 0) AS r
    FROM pairs
),
wide AS (
    SELECT a.segment,
           a.step_no,
           a.step_name,
           coalesce(a.v, 0) AS v0,
           coalesce(b.v, 0) AS v1,
           coalesce(a.r, b.r, 0) AS r0,
           coalesce(b.r, a.r, 0) AS r1
    FROM shares AS a
    JOIN shares AS b
      ON b.segment = a.segment AND b.step_no = a.step_no AND a.period = 0 AND b.period = 1
)
SELECT segment, step_no, step_name, v0, v1, r0, r1,
       (v1 - v0) * (r0 + r1) / 2 AS mix,
       (v1 - v0) * ((r0 + r1) / 2
           - (sum(v0 * r0) OVER (PARTITION BY step_no)
              + sum(v1 * r1) OVER (PARTITION BY step_no)) / 2) AS mix_c,
       (r1 - r0) * (v0 + v1) / 2 AS rate
FROM wide
ORDER BY step_no, segment;
