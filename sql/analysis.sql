-- analysis.sql
-- Example queries on the query log (logs/query_log.db).
-- Run with:  sqlite3 logs/query_log.db < sql/analysis.sql

-- 1. Overall usage and average response time
SELECT COUNT(*)                 AS total_questions,
       ROUND(AVG(total_ms), 1)  AS avg_total_ms,
       ROUND(AVG(retrieval_ms), 1)  AS avg_retrieval_ms,
       ROUND(AVG(generation_ms), 1) AS avg_generation_ms
FROM queries;

-- 2. Share of questions the model declined (answer not in documents)
SELECT SUM(declined) AS declined,
       COUNT(*)      AS total,
       ROUND(100.0 * SUM(declined) / COUNT(*), 1) AS declined_pct
FROM queries;

-- 3. Which documents are retrieved most often (JOIN + GROUP BY)
SELECT s.source,
       COUNT(*)              AS times_retrieved,
       ROUND(AVG(s.score), 3) AS avg_similarity
FROM sources s
JOIN queries q ON q.id = s.query_id
GROUP BY s.source
ORDER BY times_retrieved DESC;

-- 4. Slowest 5 questions
SELECT id, question, total_ms
FROM queries
ORDER BY total_ms DESC
LIMIT 5;

-- 5. Questions whose best match had low similarity (subquery)
SELECT id, question, top_score
FROM queries
WHERE top_score < (SELECT AVG(top_score) FROM queries)
ORDER BY top_score;

-- 6. Rank each question by response time within each day (window function)
SELECT DATE(asked_at) AS day,
       id,
       total_ms,
       RANK() OVER (PARTITION BY DATE(asked_at) ORDER BY total_ms DESC) AS slowest_rank
FROM queries;
