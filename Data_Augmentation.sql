-- Data Augmentation & Feature Engineering Query

SELECT 
    t.*,
    -- External Data
    w.precipitation as external_precipitation,
    w.snowfall as external_snowfall,
    
    -- 0. External Events (Disasters & Transit Disruptions)
    CASE 
        WHEN date(t.datetime) IN ('2011-01-26', '2011-01-27', '2011-08-27', '2011-08-28', '2012-10-29', '2012-10-30') THEN 1 
        ELSE 0 
    END as is_disaster,

    CASE 
        WHEN date(t.datetime) IN (
            -- MLK Weekend (Track Work)
            '2011-01-14', '2011-01-15', '2011-01-16', '2011-01-17',
            -- Presidents Day (Track Work)
            '2011-02-18', '2011-02-19', '2011-02-20', '2011-02-21',
            -- Memorial Day (Track Work)
            '2011-05-27', '2011-05-28', '2011-05-29', '2011-05-30',
            -- Earthquake (Rail speed restricted)
            '2011-08-23', '2011-08-24', '2011-08-25',
            -- Red Line Work
            '2011-08-06', '2011-08-07',
            -- Orange/Red Line Work
            '2012-02-17', '2012-02-18', '2012-02-19', '2012-02-20',
            -- Derecho Storm / Orange Line Work (Merged period)
            '2012-06-29', '2012-06-30', '2012-07-01', '2012-07-02',
            -- Orange Line Work (Late Aug)
            '2012-08-24', '2012-08-25', '2012-08-26'
        ) THEN 1 
        ELSE 0 
    END as transport_disruption,

    -- 1. Basic Temporal (SQLite strftime returns strings)
    CAST(strftime('%H', t.datetime) as INTEGER) as hour,
    CAST(strftime('%m', t.datetime) as INTEGER) as month,
    CAST(strftime('%Y', t.datetime) as INTEGER) as year,
    CAST(strftime('%w', t.datetime) as INTEGER) as dayofweek, -- 0=Sunday, 6=Saturday



    -- 3. Peak Hours (7-9 or 17-19 on working days)
    CASE 
        WHEN (
            (CAST(strftime('%H', t.datetime) as INTEGER) BETWEEN 7 AND 9) OR 
            (CAST(strftime('%H', t.datetime) as INTEGER) BETWEEN 17 AND 19)
        ) AND t.workingday = 1 THEN 1
        ELSE 0
    END as is_peak,



    -- 5. Domain Interactions

    (t.windspeed * t.weather) as wind_weather,

    -- 6. Cyclical Hour (Requires sin/cos UDFs in Python)
    sin(2 * 3.14159265359 * CAST(strftime('%H', t.datetime) as INTEGER) / 24.0) as hour_sin,
    cos(2 * 3.14159265359 * CAST(strftime('%H', t.datetime) as INTEGER) / 24.0) as hour_cos

FROM 
    train t
LEFT JOIN 
    weather_external w ON date(t.datetime) = w.date
LEFT JOIN 
    holidays_external h ON date(t.datetime) = h.date;
