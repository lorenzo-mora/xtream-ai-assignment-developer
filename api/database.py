import json
from pathlib import Path
import sqlite3
from typing import Any, Dict

def create_tables(path: Path) -> None:
    try:
        with sqlite3.connect(path) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS requests (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT,
                            method TEXT,
                            path TEXT,
                            status_code INTEGER
                         )''')

            c.execute('''CREATE TABLE IF NOT EXISTS prediction (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            request_id INTEGER,
                            response_time TEXT,
                            carat REAL,
                            cut TEXT,
                            color TEXT,
                            clarity TEXT,
                            depth REAL,
                            table_val REAL,
                            x REAL,
                            y REAL,
                            z REAL,
                            predicted_value REAL,
                            model TEXT,
                            note TEXT,
                            FOREIGN KEY (request_id) REFERENCES requests (id)
                         )''')

            c.execute('''CREATE TABLE IF NOT EXISTS similarity (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            request_id INTEGER,
                            response_time TEXT,
                            carat REAL,
                            cut TEXT,
                            color TEXT,
                            clarity TEXT,
                            number_samples INTEGER,
                            method TEXT,
                            dataset_name TEXT,
                            samples TEXT,
                            note TEXT,
                            FOREIGN KEY (request_id) REFERENCES requests (id)
                         )''')

            c.execute('''CREATE TABLE IF NOT EXISTS training (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            request_id INTEGER,
                            response_time TEXT,
                            type TEXT,
                            model_id TEXT,
                            parameters TEXT,
                            dataset_name TEXT,
                            preprocessing TEXT,
                            split_size REAL,
                            evaluation TEXT,
                            training_type TEXT,
                            training_configuration TEXT,
                            data_configuration TEXT,
                            note TEXT,
                            FOREIGN KEY (request_id) REFERENCES requests (id)
                         )''')

            conn.commit() 
    except sqlite3.Error as e:
        print(f"An error occurred while creating tables: {e}")

def insert_request_response(
        db_path: Path,
        timestamp: str,
        method: str,
        path: str,
        status_code: int,
        response_time: str,
        request_body: Dict[str, Any],
        response_body: Dict[str, Any]
        ) -> None:
    try:
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute('''INSERT INTO requests (timestamp, method, path, status_code)
                        VALUES (?, ?, ?, ?)''',
                        (timestamp, method, path, status_code))
            request_id = c.lastrowid

            note = None
            if status_code != 200:
                note = response_body['detail'][0]['msg']

            if path.replace("/", "") == 'predict':
                carat = request_body.get('carat')
                cut = request_body.get('cut')
                color = request_body.get('color')
                clarity = request_body.get('clarity')
                depth = request_body.get('depth')
                table_val = request_body.get('table')
                x = request_body.get('x')
                y = request_body.get('y')
                z = request_body.get('z')
                model = request_body.get('model', "19349c13-b711-4440-b669-ed9b199ad5e3")
                predicted_value = response_body.get('predicted_value')
                c.execute('''INSERT INTO prediction (request_id, response_time, carat, cut, color, clarity, depth, table_val, x, y, z, predicted_value, model, note)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                          (request_id, response_time, carat, cut, color,
                           clarity, depth, table_val, x, y, z, predicted_value,
                           model, note))
            elif path.replace("/", "") == 'similar':
                carat = request_body.get('carat')
                cut = request_body.get('cut')
                color = request_body.get('color')
                clarity = request_body.get('clarity')
                number_samples = request_body.get('n', 5)
                method = request_body.get('method', "cosine similarity")
                dataset_name = request_body.get('dataset_name', "diamonds.csv")
                samples = json.dumps(response_body.get('samples')) if response_body.get('samples') else None
                c.execute('''INSERT INTO similarity (request_id, response_time, carat, cut, color, clarity, number_samples, method, dataset_name, samples, note)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                          (request_id, response_time, carat, cut, color, clarity,
                           number_samples, method, dataset_name, samples, note))
            elif path.replace("/", "") == 'train':
                config_metadata = response_body.get('training_config', {})
                train_id = config_metadata.get('train_id')
                model_name = config_metadata.get('model_name')
                hyperparameters = json.dumps(config_metadata.get('hyperparameters')) if config_metadata.get('hyperparameters') else "default"
                dataset_name = config_metadata.get('dataset_name')
                processing_operation = json.dumps(config_metadata.get('processing_operation')) if config_metadata.get('processing_operation') else None
                split_size = config_metadata.get('split_size')
                metrics = json.dumps(config_metadata.get('metrics')) if config_metadata.get('metrics') else None
                training_type = config_metadata.get('training_type')
                if isinstance(training_type, dict):
                    training_type = json.dumps(training_type)
                train_config_file = config_metadata.get('train_config_file')
                data_config_file = config_metadata.get('data_config_file')
                note = (response_body.get('message')
                        if not note and response_body.get('message') != "Successful training" else note)
                c.execute('''INSERT INTO training (request_id, response_time, type, model_id, parameters, dataset_name, preprocessing, split_size, evaluation, training_type, training_configuration, data_configuration, note)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                          (request_id, response_time, model_name, train_id,
                           hyperparameters, dataset_name, processing_operation,
                           split_size, metrics, training_type, train_config_file,
                           data_config_file, note))

            conn.commit()
    except sqlite3.Error as e:
        print(f"An error occurred while logging request and response: {e}")