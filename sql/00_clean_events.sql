-- Clean the raw event feed.
--
-- Input : raw_events(event_id, user_id, event_name, event_time, platform, source, plan, page)
--         event_time is ISO 8601 text; a value without an offset is read as UTC.
-- Output: clean_events  one row per distinct event_id, timestamps as UTC, nothing before the
--                       user's first signup
--         users         one row per user with a signup: first signup time, cohort week
--                       (ISO, Monday start, UTC) and the segment attributes of that signup
--         clean_stats   row counts so every dropped row is accounted for
--
-- A user's first signup is the earliest `signup` event. Users with no signup have no cohort
-- and are dropped with all of their events.

CREATE OR REPLACE TABLE events_parsed AS
SELECT event_id,
       user_id,
       event_name,
       timezone('UTC', CAST(event_time AS TIMESTAMPTZ)) AS ts,
       platform,
       source,
       plan,
       page
FROM raw_events;

-- Exact duplicates share an event_id; keep one deterministic row per id.
CREATE OR REPLACE TABLE events_dedup AS
SELECT *
FROM events_parsed
QUALIFY row_number() OVER (
    PARTITION BY event_id
    ORDER BY ts, user_id, event_name, platform, source, plan, page
) = 1;

CREATE OR REPLACE TABLE users AS
SELECT user_id,
       ts AS signup_ts,
       platform,
       source,
       CAST(date_trunc('week', ts) AS DATE) AS cohort_start,
       printf('%04d-W%02d', isoyear(ts), weekofyear(ts)) AS cohort_week
FROM events_dedup
WHERE event_name = 'signup'
QUALIFY row_number() OVER (PARTITION BY user_id ORDER BY ts, event_id) = 1;

CREATE OR REPLACE TABLE clean_events AS
SELECT e.event_id, e.user_id, e.event_name, e.ts, e.platform, e.source, e.plan, e.page
FROM events_dedup AS e
JOIN users AS u ON u.user_id = e.user_id
WHERE e.ts >= u.signup_ts;

CREATE OR REPLACE TABLE clean_stats AS
SELECT
    (SELECT count(*) FROM raw_events) AS raw_rows,
    (SELECT count(*) FROM raw_events) - (SELECT count(*) FROM events_dedup)
        AS duplicate_rows_dropped,
    (SELECT count(*) FROM events_dedup AS e
      WHERE NOT EXISTS (SELECT 1 FROM users AS u WHERE u.user_id = e.user_id))
        AS no_signup_rows_dropped,
    (SELECT count(*) FROM events_dedup) - (SELECT count(*) FROM clean_events)
      - (SELECT count(*) FROM events_dedup AS e
          WHERE NOT EXISTS (SELECT 1 FROM users AS u WHERE u.user_id = e.user_id))
        AS before_signup_rows_dropped,
    (SELECT count(*) FROM clean_events) AS kept_rows;

DROP TABLE events_parsed;
DROP TABLE events_dedup;
