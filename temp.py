import sqlite3
import csv

# Create table for train.csv
with sqlite3.connect("bike_sharing.db") as conn:
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS bike_sharing_train (datetime DATETIME, season INTEGER, holiday INTEGER, workingday INTEGER, temp REAL, atemp REAL, weather INTEGER, humidity INTEGER, windspeed REAL, casual INTEGER, registered INTEGER, count INTEGER)")
    conn.commit()

# Create table for test.csv
with sqlite3.connect("bike_sharing.db") as conn:
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS bike_sharing_test (datetime DATETIME, season INTEGER,  holiday INTEGER, workingday INTEGER, temp REAL, atemp REAL, weather INTEGER, humidity INTEGER, windspeed REAL)")
    conn.commit()

# Load data from train.csv and test.csv into sqlite3 database
with sqlite3.connect("bike_sharing.db") as conn:
    with open("bike-sharing-demand\\train.csv", "r") as f:
        cursor = conn.cursor()
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            cursor.execute("INSERT INTO bike_sharing_train VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)
    
    with open("bike-sharing-demand\\test.csv", "r") as f:
        cursor = conn.cursor()
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            cursor.execute("INSERT INTO bike_sharing_test VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", row)
    
    conn.commit()
    
