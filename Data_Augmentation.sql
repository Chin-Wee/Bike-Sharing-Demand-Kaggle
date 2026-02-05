-- Data Augmentation Query
-- Joins the main training data with external weather and holiday datasets

SELECT 
    t.*,
    w.precipitation as external_precipitation,
    w.snowfall as external_snowfall,
    h.holiday_name as specific_holiday_name,
    CASE WHEN h.holiday_name IS NOT NULL THEN 1 ELSE 0 END as is_federal_holiday
FROM 
    train t
LEFT JOIN 
    weather_external w ON date(t.datetime) = w.date
LEFT JOIN 
    holidays_external h ON date(t.datetime) = h.date;
