-- Pure Data Retrieval & Join Query (Logic moved to Python)

SELECT 
    t.*,
    -- Weather External Data
    w.precipitation as external_precipitation,
    w.snowfall as external_snowfall,
    
    -- Holiday Data
    h.holiday_name as specific_holiday_name,
    
    -- Disruption Data (Joined from CSV)
    d.event_type as disruption_event_type,
    d.description as disruption_description

FROM 
    train t
LEFT JOIN 
    weather_external w ON date(t.datetime) = w.date
LEFT JOIN 
    holidays_external h ON date(t.datetime) = h.date
LEFT JOIN
    disruptions d ON date(t.datetime) = d.date;
